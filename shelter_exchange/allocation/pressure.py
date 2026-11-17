from __future__ import annotations

import datetime as dt
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..admission.eligibility import EligibilityPolicy
from ..contracts import ContractError
from ..inventory.allocator import BedAllocator, requirements_for
from ..inventory.calendar import InventoryCalendar, NightRange

# 占用床夜的预留状态(与 InventoryCalendar.occupied 口径一致)
OCCUPYING_STATUSES = frozenset({'held', 'confirmed', 'checked_in'})
# 情景中允许释放的预留状态(未确认预留/已确认未入住, 在住不可释放)
RELEASABLE_STATUSES = frozenset({'held', 'confirmed'})
# 未确认预留
UNCONFIRMED_STATUS = 'held'

CLASSIFICATION_LABELS = {
    'ineligible': '资格不合格',
    'accommodatable': '可入住',
    'waitlisted': '候补',
    'allocated': '已分配(沿用已有决定)',
}
NIGHT_STATUS_LABELS = {
    'available': '可安排',
    'bed_shortage': '床位不足',
    'room_type_mismatch': '房型不符',
    'held_reservation': '已预留但尚未入住',
}
SUPPLY_LABELS = {'held': '未确认预留', 'confirmed': '已确认未入住', 'checked_in': '在住'}

DEFAULT_TEMPORARY_BED = {'room_gender': 'any', 'accessible': False, 'commute_minutes': 999, 'station_priority': 99}


class ScenarioError(ValueError):
    pass


@dataclass(frozen=True)
class Scenario:
    """情景假设: 调整站点容量(协调临时床位)、释放预留或停用床位。

    capacity_adjustments: {station_id: 新增临时床位数}, 临时床位属性由 temporary_bed_template 控制
    release_reservation_ids: 视为已释放的预留(仅 held/confirmed 可释放)
    remove_bed_ids: 停用床位(容量调减的显式口径)
    """
    name: str
    capacity_adjustments: Mapping[str, int] = field(default_factory=dict)
    release_reservation_ids: tuple[str, ...] = ()
    remove_bed_ids: tuple[str, ...] = ()
    temporary_bed_template: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ScenarioError('scenario name required')
        object.__setattr__(self, 'release_reservation_ids', tuple(str(item) for item in self.release_reservation_ids))
        object.__setattr__(self, 'remove_bed_ids', tuple(str(item) for item in self.remove_bed_ids))
        for station_id, count in self.capacity_adjustments.items():
            if not isinstance(count, int) or count < 0:
                raise ScenarioError(f'capacity adjustment for {station_id} must be a non-negative integer; use remove_bed_ids to reduce capacity')


class PressureAnalyzer:
    """候补申请 × 床夜库存的站点/日期粒度供需压力分析(只读、顾问式)。

    日历口径与现有分配一致: 住宿为半开区间 [arrival, departure), 退住日当天不占用,
    退住日与入住日相邻的预留互不冲突。分析不写入任何记录、不触发通知, 不推翻已有的
    分配与通知决定; 情景试算按候补顺序逐个兑现床夜, 同一床夜不会对相交申请重复承诺。
    """

    def __init__(self, calendar: InventoryCalendar | None = None, eligibility: EligibilityPolicy | None = None):
        self.calendar = calendar or InventoryCalendar()
        self.eligibility = eligibility or EligibilityPolicy()
        self.allocator = BedAllocator(self.calendar)

    # ------------------------------------------------------------------ public

    def analyze(self, applications: Sequence[Mapping[str, Any]], beds: Sequence[Mapping[str, Any]],
                reservations: Sequence[Mapping[str, Any]], horizon: NightRange,
                prior_stays: Sequence[Mapping[str, Any]] = (), today: dt.date | None = None,
                scenario: Scenario | None = None) -> dict[str, Any]:
        today = today or dt.date.today()
        beds, reservations = self._apply_scenario(beds, reservations, scenario)
        reservations = self._with_committed_allocations(applications, reservations)
        occupancy, skipped_invalid = self._occupancy_index(reservations)
        entries = [self._classify(application, beds, reservations, occupancy, prior_stays, today) for application in applications]
        grid = self._station_grid(horizon, beds, occupancy, entries)
        simulation = self._simulate(entries, beds, reservations)
        report = {
            'kind': 'pressure_report',
            'advisory': True,
            'calendar': 'half_open_nights:[arrival,departure)',
            'horizon': {'start': horizon.arrival.isoformat(), 'end': horizon.departure.isoformat(), 'nights': horizon.nights},
            'labels': {'classification': dict(CLASSIFICATION_LABELS), 'night_status': dict(NIGHT_STATUS_LABELS), 'supply': dict(SUPPLY_LABELS)},
            'summary': self._summary(entries, grid, simulation, reservations, occupancy, horizon, skipped_invalid),
            'applications': [entry['report'] for entry in entries],
            'stations': grid,
            'simulation': simulation['report'],
        }
        if scenario is not None:
            report['scenario'] = {
                'name': scenario.name,
                'capacity_adjustments': dict(sorted(scenario.capacity_adjustments.items())),
                'release_reservation_ids': sorted(scenario.release_reservation_ids),
                'remove_bed_ids': sorted(scenario.remove_bed_ids),
            }
        return report

    def compare(self, base: Mapping[str, Any], scenarios: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """把若干情景报告与基准报告对比, 输出指标差量、申请分类迁移与站点-日期差量。"""
        base_metrics = self._metrics(base)
        base_class = {row['application_id']: row['classification'] for row in base['applications']}
        base_promoted = {row['application_id'] for row in base['simulation']['promotable']}
        base_cells = {(station['station_id'], cell['night']): cell for station in base['stations'] for cell in station['nights']}
        result = {'base': base_metrics, 'scenarios': []}
        for report in scenarios:
            metrics = self._metrics(report)
            classes = {row['application_id']: row['classification'] for row in report['applications']}
            promoted = {row['application_id'] for row in report['simulation']['promotable']}
            transitions = [
                {'application_id': application_id, 'from': base_class.get(application_id), 'to': classes.get(application_id)}
                for application_id in sorted(set(base_class) | set(classes))
                if base_class.get(application_id) != classes.get(application_id)
            ]
            cells = {(station['station_id'], cell['night']): cell for station in report['stations'] for cell in station['nights']}
            deltas = []
            for key in sorted(set(base_cells) | set(cells)):
                before, after = base_cells.get(key), cells.get(key)
                delta_free = (after['free'] if after else 0) - (before['free'] if before else 0)
                delta_unmet = (len(after['unmet_application_ids']) if after else 0) - (len(before['unmet_application_ids']) if before else 0)
                if delta_free or delta_unmet:
                    deltas.append({'station_id': key[0], 'night': key[1], 'delta_free': delta_free, 'delta_unmet': delta_unmet})
            result['scenarios'].append({
                'name': report.get('scenario', {}).get('name'),
                'metrics': metrics,
                'delta': {name: metrics[name] - base_metrics[name] for name in metrics},
                'resolved_application_ids': sorted(promoted - base_promoted),
                'classification_transitions': transitions,
                'station_deltas': deltas,
            })
        return result

    # ------------------------------------------------------------- classify

    def _classify(self, application: Mapping[str, Any], beds: Sequence[Mapping[str, Any]],
                  reservations: Sequence[Mapping[str, Any]], occupancy: Mapping[str, Any],
                  prior_stays: Sequence[Mapping[str, Any]], today: dt.date) -> dict[str, Any]:
        application_id = str(application.get('application_id', ''))
        applicant_id = str(application.get('applicant_id') or application.get('owner_id') or '')
        attributes = application.get('attributes') if isinstance(application.get('attributes'), Mapping) else {}
        recorded = attributes.get('state')
        arrival = self._parse_date(application.get('arrival') or application.get('effective_from'))
        departure = self._parse_date(application.get('departure') or application.get('effective_to'))
        period = NightRange(arrival, departure) if arrival and departure and departure > arrival else None
        payload = dict(application)
        if arrival:
            payload['arrival'] = arrival.isoformat()
        if departure:
            payload['departure'] = departure.isoformat()
        payload['applicant_id'] = applicant_id

        reasons: tuple[str, ...] = ()
        approved_nights = 0
        continuous = False
        nights: list[dict[str, Any]] = []
        compatible: list[Mapping[str, Any]] = []
        if recorded in {'rejected', 'allocated'}:
            # 已有决定(含已通知结果)原样沿用, 分析不重新裁定
            classification = 'ineligible' if recorded == 'rejected' else 'allocated'
            decision_source = 'recorded'
            reasons = tuple(attributes.get('reasons') or ())
            approved_nights = int(attributes.get('approved_nights') or 0)
        else:
            decision_source = 'evaluated'
            try:
                decision = self.eligibility.evaluate(payload, prior_stays, today)
            except ContractError:
                decision = None
            if decision is None:
                classification, reasons = 'ineligible', ('invalid_application',)
            elif not decision.eligible:
                classification, reasons = 'ineligible', decision.reasons
                approved_nights = decision.approved_nights
            elif period is None:
                classification, reasons = 'ineligible', ('invalid_period',)
            else:
                approved_nights = decision.approved_nights
                compatible = self.calendar.compatible_beds(beds, requirements_for(payload))
                continuous = self.allocator.allocate(payload, beds, reservations).status == 'allocated'
                classification = 'accommodatable' if continuous else 'waitlisted'
                nights = self._diagnose_nights(period, compatible, occupancy)

        gap_nights = [row['night'] for row in nights if row['status'] != 'available']
        report = {
            'application_id': application_id,
            'applicant_id': applicant_id,
            'classification': classification,
            'classification_label': CLASSIFICATION_LABELS[classification],
            'recorded_state': recorded,
            'decision_source': decision_source,
            'eligibility_reasons': list(reasons),
            'requested_nights': period.nights if period else 0,
            'approved_nights': approved_nights,
            'continuous_bed_available': continuous,
            'fragmented': classification == 'waitlisted' and not gap_nights and bool(nights),
            'compatible_bed_ids': sorted(str(bed['bed_id']) for bed in compatible),
            'compatible_station_ids': sorted({str(bed.get('station_id')) for bed in compatible if bed.get('station_id')}),
            'nights': nights,
            'gap_nights': gap_nights,
            'releasable_nights': [
                {'night': row['night'], 'reservation_ids': row['releasable_reservation_ids']}
                for row in nights if row['status'] == 'held_reservation'
            ],
        }
        if classification == 'allocated':
            report['allocated_bed_id'] = attributes.get('bed_id')
            report['allocated_station_id'] = attributes.get('station_id')
        return {
            'application_id': application_id,
            'classification': classification,
            'night_set': {row['night'] for row in nights},
            'compatible': compatible,
            'payload': payload,
            'report': report,
        }

    def _diagnose_nights(self, period: NightRange, compatible: Sequence[Mapping[str, Any]],
                         occupancy: Mapping[str, Any]) -> list[dict[str, Any]]:
        nights = []
        for night in period.split():
            free_ids, blocking_ids, releasable_ids = [], set(), set()
            for bed in compatible:
                blockers = occupancy.get(str(bed['bed_id']), {}).get(night, ())
                if not blockers:
                    free_ids.append(str(bed['bed_id']))
                    continue
                blocking_ids.update(str(row.get('reservation_id')) for row in blockers)
                if all(row.get('status') == UNCONFIRMED_STATUS for row in blockers):
                    # 该床夜的阻挡全部来自未确认预留, 释放后本晚即可安排
                    releasable_ids.update(str(row.get('reservation_id')) for row in blockers)
            if free_ids:
                status = 'available'
            elif not compatible:
                status = 'room_type_mismatch'
            elif releasable_ids:
                status = 'held_reservation'
            else:
                status = 'bed_shortage'
            nights.append({
                'night': night.isoformat(),
                'status': status,
                'status_label': NIGHT_STATUS_LABELS[status],
                'free_bed_ids': sorted(free_ids),
                'blocking_reservation_ids': sorted(blocking_ids),
                'releasable_reservation_ids': sorted(releasable_ids),
            })
        return nights

    # ---------------------------------------------------------------- grid

    def _station_grid(self, horizon: NightRange, beds: Sequence[Mapping[str, Any]],
                      occupancy: Mapping[str, Any], entries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        stations = sorted({str(bed.get('station_id')) for bed in beds if bed.get('station_id')})
        waiting = [entry for entry in entries if entry['classification'] == 'waitlisted']
        grid = []
        for station in stations:
            active_beds = [bed for bed in beds if str(bed.get('station_id')) == station and bed.get('status') == 'active']
            cells = []
            for night in horizon.split():
                free_ids, confirmed_ids, checked_in_ids, held_ids = [], set(), set(), set()
                for bed in active_beds:
                    blockers = occupancy.get(str(bed['bed_id']), {}).get(night, ())
                    if not blockers:
                        free_ids.append(str(bed['bed_id']))
                    for row in blockers:
                        bucket = {'confirmed': confirmed_ids, 'checked_in': checked_in_ids, 'held': held_ids}.get(row.get('status'))
                        if bucket is not None:
                            bucket.add(str(row.get('reservation_id')))
                demand_ids, unmet_ids = [], []
                for entry in waiting:
                    if night.isoformat() not in entry['night_set']:
                        continue
                    here = [bed for bed in entry['compatible'] if str(bed.get('station_id')) == station]
                    if not here:
                        continue
                    demand_ids.append(entry['application_id'])
                    if not any(not occupancy.get(str(bed['bed_id']), {}).get(night) for bed in here):
                        unmet_ids.append(entry['application_id'])
                cells.append({
                    'night': night.isoformat(),
                    'capacity': len(active_beds),
                    'free': len(free_ids),
                    'free_bed_ids': sorted(free_ids),
                    'confirmed': len(confirmed_ids),
                    'confirmed_reservation_ids': sorted(confirmed_ids),
                    'checked_in': len(checked_in_ids),
                    'checked_in_reservation_ids': sorted(checked_in_ids),
                    'held': len(held_ids),
                    'held_reservation_ids': sorted(held_ids),
                    'releasable_bed_nights': len(held_ids),
                    'waitlist_application_ids': sorted(demand_ids),
                    'unmet_application_ids': sorted(unmet_ids),
                })
            totals = {
                'capacity_bed_nights': sum(cell['capacity'] for cell in cells),
                'free_bed_nights': sum(cell['free'] for cell in cells),
                'confirmed_bed_nights': sum(cell['confirmed'] for cell in cells),
                'checked_in_bed_nights': sum(cell['checked_in'] for cell in cells),
                'held_bed_nights': sum(cell['held'] for cell in cells),
                'releasable_bed_nights': sum(cell['releasable_bed_nights'] for cell in cells),
                'waitlist_requests': sum(len(cell['waitlist_application_ids']) for cell in cells),
                'unmet_requests': sum(len(cell['unmet_application_ids']) for cell in cells),
            }
            grid.append({'station_id': station, 'nights': cells, 'totals': totals})
        return grid

    # ------------------------------------------------------------ simulate

    def _simulate(self, entries: Sequence[Mapping[str, Any]], beds: Sequence[Mapping[str, Any]],
                  reservations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        queue = [entry['payload'] for entry in entries if entry['classification'] in {'accommodatable', 'waitlisted'}]
        working = [dict(row) for row in reservations]
        before = len(working)
        # promote_waitlist 按候补顺序逐个兑现并即时占用床夜, 相交申请不会拿到同一床夜
        allocations = self.allocator.promote_waitlist(queue, beds, working)
        provisional_ids = {str(row['reservation_id']) for row in working[before:]}
        report = {
            'queue_order': [allocation.application_id for allocation in allocations],
            'promotable': [
                {'application_id': allocation.application_id, 'bed_id': allocation.bed_id, 'station_id': allocation.station_id}
                for allocation in allocations if allocation.bed_id
            ],
            'still_waiting': sorted(allocation.application_id for allocation in allocations if not allocation.bed_id),
            'provisional_reservation_ids': sorted(provisional_ids),
            'integrity': {'double_promises': self._double_promises(working, provisional_ids)},
            'note': '试算结果仅用于评估, 不写入库存也不触发通知',
        }
        return {'report': report, 'promoted_ids': {row['application_id'] for row in report['promotable']}}

    def _double_promises(self, reservations: Sequence[Mapping[str, Any]], provisional_ids: set[str]) -> list[dict[str, Any]]:
        occupancy, _ = self._occupancy_index(reservations)
        conflicts = []
        for bed_id, nights in occupancy.items():
            for night, rows in nights.items():
                ids = sorted({str(row.get('reservation_id')) for row in rows})
                if len(ids) > 1 and provisional_ids & set(ids):
                    conflicts.append({'bed_id': bed_id, 'night': night.isoformat(), 'reservation_ids': ids})
        return conflicts

    # ------------------------------------------------------------ scenario

    def _apply_scenario(self, beds: Sequence[Mapping[str, Any]], reservations: Sequence[Mapping[str, Any]],
                        scenario: Scenario | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        beds = [dict(bed) for bed in beds]
        reservations = [dict(row) for row in reservations]
        if scenario is None:
            return beds, reservations
        if scenario.release_reservation_ids:
            by_id = {str(row.get('reservation_id')): row for row in reservations}
            unknown = sorted(set(scenario.release_reservation_ids) - set(by_id))
            if unknown:
                raise ScenarioError(f'unknown reservations: {unknown}')
            for reservation_id in scenario.release_reservation_ids:
                row = by_id[reservation_id]
                if row.get('status') not in RELEASABLE_STATUSES:
                    raise ScenarioError(f"reservation {reservation_id} with status {row.get('status')} cannot be released")
                row['status'] = 'released'
        if scenario.remove_bed_ids:
            known = {str(bed.get('bed_id')) for bed in beds}
            unknown = sorted(set(scenario.remove_bed_ids) - known)
            if unknown:
                raise ScenarioError(f'unknown beds: {unknown}')
            for bed in beds:
                if str(bed.get('bed_id')) in set(scenario.remove_bed_ids):
                    bed['status'] = 'inactive'
        template = {**DEFAULT_TEMPORARY_BED, **dict(scenario.temporary_bed_template)}
        for station_id, count in sorted(scenario.capacity_adjustments.items()):
            for index in range(count):
                beds.append({**template, 'bed_id': f'temporary:{station_id}:{index + 1}', 'station_id': station_id, 'status': 'active', 'temporary': True})
        return beds, reservations

    # ------------------------------------------------------------- helpers

    def _with_committed_allocations(self, applications: Sequence[Mapping[str, Any]],
                                    reservations: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """已有"已分配"决定但尚无对应预留的, 补一条承诺占用, 防止同一床夜被再次承诺。"""
        result = [dict(row) for row in reservations]
        for application in applications:
            attributes = application.get('attributes') if isinstance(application.get('attributes'), Mapping) else {}
            if attributes.get('state') != 'allocated' or not attributes.get('bed_id'):
                continue
            bed_id = str(attributes['bed_id'])
            arrival = self._parse_date(application.get('arrival') or application.get('effective_from'))
            departure = self._parse_date(application.get('departure') or application.get('effective_to'))
            if not arrival or not departure or departure <= arrival:
                continue
            period = NightRange(arrival, departure)
            if any(str(row.get('bed_id')) == bed_id and row.get('status') in OCCUPYING_STATUSES and self._overlaps(row, period) for row in result):
                continue
            result.append({
                'reservation_id': f"committed:{application.get('application_id')}",
                'bed_id': bed_id,
                'arrival': arrival.isoformat(),
                'departure': departure.isoformat(),
                'status': 'confirmed',
                'source': 'recorded_allocation',
            })
        return result

    def _occupancy_index(self, reservations: Sequence[Mapping[str, Any]]) -> tuple[dict[str, dict[dt.date, list[Mapping[str, Any]]]], int]:
        index: dict[str, dict[dt.date, list[Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(list))
        skipped = 0
        for row in reservations:
            if row.get('status') not in OCCUPYING_STATUSES:
                continue
            arrival, departure = self._parse_date(row.get('arrival')), self._parse_date(row.get('departure'))
            if not arrival or not departure or departure <= arrival:
                skipped += 1
                continue
            for night in NightRange(arrival, departure).split():
                index[str(row.get('bed_id'))][night].append(row)
        return index, skipped

    def _summary(self, entries: Sequence[Mapping[str, Any]], grid: Sequence[Mapping[str, Any]],
                 simulation: Mapping[str, Any], reservations: Sequence[Mapping[str, Any]],
                 occupancy: Mapping[str, Any], horizon: NightRange, skipped_invalid: int) -> dict[str, Any]:
        counts = Counter(entry['classification'] for entry in entries)
        horizon_nights = set(horizon.split())
        gap_nights = 0
        for entry in entries:
            if entry['classification'] != 'waitlisted':
                continue
            gap_nights += sum(1 for night in entry['report']['gap_nights'] if dt.date.fromisoformat(night) in horizon_nights)
        held_ids = sorted({str(row.get('reservation_id')) for row in reservations if row.get('status') == UNCONFIRMED_STATUS})
        releasable = sum(
            1
            for nights in occupancy.values() for night, rows in nights.items()
            if night in horizon_nights for row in rows if row.get('status') == UNCONFIRMED_STATUS
        )
        return {
            'applications_total': len(entries),
            'by_classification': {name: counts.get(name, 0) for name in ('ineligible', 'accommodatable', 'waitlisted', 'allocated')},
            'waitlist_gap_nights': gap_nights,
            'unmet_bed_nights': sum(len(cell['unmet_application_ids']) for station in grid for cell in station['nights']),
            'releasable_bed_nights': releasable,
            'unconfirmed_reservations': len(held_ids),
            'unconfirmed_reservation_ids': held_ids,
            'promotable_now': len(simulation['report']['promotable']),
            'still_waiting': len(simulation['report']['still_waiting']),
            'skipped_invalid_reservations': skipped_invalid,
        }

    @staticmethod
    def _metrics(report: Mapping[str, Any]) -> dict[str, int]:
        summary = report['summary']
        return {
            'waitlisted': summary['by_classification']['waitlisted'],
            'accommodatable': summary['by_classification']['accommodatable'],
            'promotable': summary['promotable_now'],
            'gap_nights': summary['waitlist_gap_nights'],
            'unmet_bed_nights': summary['unmet_bed_nights'],
        }

    @staticmethod
    def _overlaps(reservation: Mapping[str, Any], period: NightRange) -> bool:
        arrival, departure = PressureAnalyzer._parse_date(reservation.get('arrival')), PressureAnalyzer._parse_date(reservation.get('departure'))
        return bool(arrival and departure and departure > arrival and NightRange(arrival, departure).overlaps(period))

    @staticmethod
    def _parse_date(value: Any) -> dt.date | None:
        if isinstance(value, dt.datetime):
            return value.date()
        if isinstance(value, dt.date):
            return value
        if isinstance(value, str):
            try:
                return dt.date.fromisoformat(value)
            except ValueError:
                return None
        return None
