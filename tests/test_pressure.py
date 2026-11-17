import copy
import datetime as dt
import json
import tempfile
import unittest
from pathlib import Path

from shelter_exchange.allocation.pressure import PressureAnalyzer, Scenario, ScenarioError
from shelter_exchange.inventory.calendar import NightRange
from shelter_exchange.operations.service import ShelterService
from shelter_exchange.storage import Database

TODAY = dt.date(2026, 11, 17)
HORIZON = NightRange(dt.date(2026, 11, 19), dt.date(2026, 11, 24))


def bed(bed_id, station='s1', gender='any', accessible=False, status='active', commute=10):
    return {'bed_id': bed_id, 'station_id': station, 'status': status, 'room_gender': gender,
            'accessible': accessible, 'commute_minutes': commute}


def res(reservation_id, bed_id, arrival, departure, status='confirmed'):
    return {'reservation_id': reservation_id, 'bed_id': bed_id, 'arrival': arrival, 'departure': departure, 'status': status}


def app(application_id, arrival='2026-11-20', departure='2026-11-22', gender='any', accessible=False,
        priority=0, submitted_at='2026-11-16T09:00:00+00:00', **extra):
    payload = {'application_id': application_id, 'applicant_id': f'person-{application_id}',
               'arrival': arrival, 'departure': departure, 'graduation_year': 2026,
               'interview_invitation': 'invite-123456', 'diploma_reference': 'degree-123',
               'gender': gender, 'accessible': accessible, 'priority': priority, 'submitted_at': submitted_at}
    payload.update(extra)
    return payload


def analyze(applications, beds, reservations, horizon=HORIZON, scenario=None):
    return PressureAnalyzer().analyze(applications, beds, reservations, horizon, today=TODAY, scenario=scenario)


def entry(report, application_id):
    return next(row for row in report['applications'] if row['application_id'] == application_id)


def cell(report, station_id, night):
    station = next(row for row in report['stations'] if row['station_id'] == station_id)
    return next(row for row in station['nights'] if row['night'] == night)


class ClassificationTests(unittest.TestCase):
    def test_split_ineligible_accommodatable_waitlisted(self):
        beds = [bed('b1'), bed('b2')]
        reservations = [res('r1', 'b2', '2026-11-19', '2026-11-25'), res('r2', 'b1', '2026-11-22', '2026-11-23')]
        applications = [app('bad', interview_invitation=''), app('ok'), app('wait', departure='2026-11-23')]
        report = analyze(applications, beds, reservations)
        self.assertEqual(entry(report, 'bad')['classification'], 'ineligible')
        self.assertEqual(entry(report, 'ok')['classification'], 'accommodatable')
        self.assertEqual(entry(report, 'wait')['classification'], 'waitlisted')
        self.assertEqual(report['summary']['by_classification'],
                         {'ineligible': 1, 'accommodatable': 1, 'waitlisted': 1, 'allocated': 0})

    def test_labels_cover_required_categories(self):
        report = analyze([], [], [])
        labels = report['labels']['classification']
        self.assertEqual(labels['ineligible'], '资格不合格')
        self.assertEqual(labels['accommodatable'], '可入住')
        self.assertEqual(labels['waitlisted'], '候补')
        self.assertEqual(report['labels']['supply']['held'], '未确认预留')

    def test_ineligible_application_is_not_demand(self):
        report = analyze([app('bad', interview_invitation='')], [bed('b1')], [])
        self.assertEqual(report['stations'][0]['totals']['waitlist_requests'], 0)
        self.assertEqual(report['simulation']['queue_order'], [])


class NightDiagnosisTests(unittest.TestCase):
    def test_bed_shortage_when_blocker_confirmed(self):
        beds = [bed('b1')]
        reservations = [res('r1', 'b1', '2026-11-20', '2026-11-22')]
        row = entry(analyze([app('a1')], beds, reservations), 'a1')
        self.assertEqual([night['status'] for night in row['nights']], ['bed_shortage', 'bed_shortage'])
        self.assertEqual(row['nights'][0]['blocking_reservation_ids'], ['r1'])
        self.assertEqual(row['releasable_nights'], [])

    def test_held_reservation_is_releasable(self):
        beds = [bed('b1')]
        reservations = [res('r1', 'b1', '2026-11-20', '2026-11-22', status='held')]
        row = entry(analyze([app('a1')], beds, reservations), 'a1')
        self.assertEqual([night['status'] for night in row['nights']], ['held_reservation', 'held_reservation'])
        self.assertEqual(row['releasable_nights'],
                         [{'night': '2026-11-20', 'reservation_ids': ['r1']},
                          {'night': '2026-11-21', 'reservation_ids': ['r1']}])

    def test_room_type_mismatch(self):
        beds = [bed('b1', gender='male')]
        row = entry(analyze([app('a1', gender='female')], beds, []), 'a1')
        self.assertEqual({night['status'] for night in row['nights']}, {'room_type_mismatch'})
        self.assertEqual(row['compatible_bed_ids'], [])

    def test_accessible_requirement_mismatch(self):
        beds = [bed('b1', accessible=False)]
        row = entry(analyze([app('a1', accessible=True)], beds, []), 'a1')
        self.assertEqual({night['status'] for night in row['nights']}, {'room_type_mismatch'})

    def test_adjacent_departure_and_arrival_do_not_conflict(self):
        # 退住日与入住日相邻: 按现有半开区间口径, 互不冲突
        beds = [bed('b1')]
        reservations = [res('r1', 'b1', '2026-11-18', '2026-11-20')]
        row = entry(analyze([app('a1')], beds, reservations), 'a1')
        self.assertEqual(row['classification'], 'accommodatable')
        self.assertEqual([night['status'] for night in row['nights']], ['available', 'available'])

    def test_gap_nights_across_multi_night_stay(self):
        beds = [bed('b1')]
        reservations = [res('r1', 'b1', '2026-11-21', '2026-11-22')]
        row = entry(analyze([app('a1', departure='2026-11-23')], beds, reservations), 'a1')
        self.assertEqual(row['classification'], 'waitlisted')
        self.assertEqual(row['gap_nights'], ['2026-11-21'])
        self.assertFalse(row['fragmented'])

    def test_fragmented_when_each_night_free_on_different_bed(self):
        beds = [bed('b1'), bed('b2')]
        reservations = [res('r1', 'b1', '2026-11-21', '2026-11-22'), res('r2', 'b2', '2026-11-20', '2026-11-21')]
        row = entry(analyze([app('a1')], beds, reservations), 'a1')
        self.assertEqual(row['classification'], 'waitlisted')
        self.assertEqual(row['gap_nights'], [])
        self.assertTrue(row['fragmented'])
        self.assertFalse(row['continuous_bed_available'])


class StationGridTests(unittest.TestCase):
    def test_cell_counts_and_traceability(self):
        beds = [bed('b1', 's1'), bed('b2', 's1'), bed('b3', 's2', accessible=True)]
        reservations = [res('r1', 'b1', '2026-11-20', '2026-11-22', status='held'),
                        res('r2', 'b2', '2026-11-20', '2026-11-21', status='checked_in')]
        applications = [app('w1', station_ids=['s1'])]
        report = analyze(applications, beds, reservations)
        target = cell(report, 's1', '2026-11-20')
        self.assertEqual(target['capacity'], 2)
        self.assertEqual(target['free'], 0)
        self.assertEqual(target['held'], 1)
        self.assertEqual(target['held_reservation_ids'], ['r1'])
        self.assertEqual(target['checked_in_reservation_ids'], ['r2'])
        self.assertEqual(target['releasable_bed_nights'], 1)
        self.assertEqual(target['waitlist_application_ids'], ['w1'])
        self.assertEqual(target['unmet_application_ids'], ['w1'])
        # 申请只兼容 s1, 不计入 s2 的需求
        self.assertEqual(cell(report, 's2', '2026-11-20')['waitlist_application_ids'], [])
        # 汇总可追溯到明细
        station = next(row for row in report['stations'] if row['station_id'] == 's1')
        self.assertEqual(station['totals']['held_bed_nights'], sum(c['held'] for c in station['nights']))
        self.assertEqual(station['totals']['unmet_requests'], sum(len(c['unmet_application_ids']) for c in station['nights']))

    def test_summary_counts(self):
        beds = [bed('b1')]
        reservations = [res('r1', 'b1', '2026-11-20', '2026-11-22', status='held')]
        report = analyze([app('w1')], beds, reservations)
        summary = report['summary']
        self.assertEqual(summary['unconfirmed_reservations'], 1)
        self.assertEqual(summary['unconfirmed_reservation_ids'], ['r1'])
        self.assertEqual(summary['releasable_bed_nights'], 2)
        self.assertEqual(summary['waitlist_gap_nights'], 2)
        self.assertEqual(summary['unmet_bed_nights'], 2)

    def test_horizon_clips_grid_but_not_application_nights(self):
        beds = [bed('b1')]
        report = analyze([app('a1', arrival='2026-11-22', departure='2026-11-30')], beds, [])
        row = entry(report, 'a1')
        self.assertEqual(row['requested_nights'], 8)
        self.assertEqual(len(row['nights']), 8)
        self.assertEqual(len(report['stations'][0]['nights']), HORIZON.nights)


class SimulationTests(unittest.TestCase):
    def test_no_double_promise_for_overlapping_applications(self):
        beds = [bed('b1')]
        applications = [app('low', priority=1), app('high', priority=5)]
        report = analyze(applications, beds, [])
        simulation = report['simulation']
        self.assertEqual(simulation['queue_order'], ['high', 'low'])
        self.assertEqual([row['application_id'] for row in simulation['promotable']], ['high'])
        self.assertEqual(simulation['still_waiting'], ['low'])
        self.assertEqual(simulation['integrity']['double_promises'], [])

    def test_simulation_is_advisory_only(self):
        report = analyze([app('a1')], [bed('b1')], [])
        self.assertIn('不写入库存', report['simulation']['note'])
        self.assertTrue(report['advisory'])


class ScenarioTests(unittest.TestCase):
    def test_release_held_reservation_resolves_waitlist(self):
        beds = [bed('b1')]
        reservations = [res('r1', 'b1', '2026-11-20', '2026-11-22', status='held')]
        base = analyze([app('w1')], beds, reservations)
        self.assertEqual(base['summary']['promotable_now'], 0)
        scenario = analyze([app('w1')], beds, reservations, scenario=Scenario('释放r1', release_reservation_ids=('r1',)))
        self.assertEqual(scenario['summary']['promotable_now'], 1)
        comparison = PressureAnalyzer().compare(base, [scenario])
        row = comparison['scenarios'][0]
        self.assertEqual(row['name'], '释放r1')
        self.assertEqual(row['delta']['promotable'], 1)
        self.assertEqual(row['resolved_application_ids'], ['w1'])
        self.assertEqual(row['classification_transitions'],
                         [{'application_id': 'w1', 'from': 'waitlisted', 'to': 'accommodatable'}])
        self.assertIn({'station_id': 's1', 'night': '2026-11-20', 'delta_free': 1, 'delta_unmet': -1}, row['station_deltas'])

    def test_capacity_adjustment_adds_temporary_beds(self):
        base = analyze([app('w1')], [], [])
        scenario = analyze([app('w1')], [], [], scenario=Scenario('协调临时床位', capacity_adjustments={'s1': 1}))
        target = cell(scenario, 's1', '2026-11-20')
        self.assertEqual(target['capacity'], 1)
        self.assertEqual(target['free_bed_ids'], ['temporary:s1:1'])
        comparison = PressureAnalyzer().compare(base, [scenario])
        self.assertEqual(comparison['scenarios'][0]['resolved_application_ids'], ['w1'])

    def test_remove_bed_ids_reduces_capacity(self):
        beds = [bed('b1')]
        scenario = analyze([], beds, [], scenario=Scenario('停用b1', remove_bed_ids=('b1',)))
        self.assertEqual(cell(scenario, 's1', '2026-11-20')['capacity'], 0)

    def test_release_rejects_checked_in_reservation(self):
        reservations = [res('r1', 'b1', '2026-11-20', '2026-11-22', status='checked_in')]
        with self.assertRaises(ScenarioError):
            analyze([], [bed('b1')], reservations, scenario=Scenario('非法释放', release_reservation_ids=('r1',)))

    def test_release_rejects_unknown_reservation(self):
        with self.assertRaises(ScenarioError):
            analyze([], [], [], scenario=Scenario('未知预留', release_reservation_ids=('nope',)))

    def test_negative_capacity_rejected(self):
        with self.assertRaises(ScenarioError):
            Scenario('负数容量', capacity_adjustments={'s1': -1})


class RecordedDecisionTests(unittest.TestCase):
    def test_recorded_rejection_is_honored(self):
        payload = app('a1')
        payload['attributes'] = {'state': 'rejected', 'reasons': ['free_night_limit_exhausted'], 'approved_nights': 0}
        row = entry(analyze([payload], [bed('b1')], []), 'a1')
        self.assertEqual(row['classification'], 'ineligible')
        self.assertEqual(row['decision_source'], 'recorded')
        self.assertEqual(row['eligibility_reasons'], ['free_night_limit_exhausted'])

    def test_recorded_allocation_commits_bed_nights(self):
        allocated = app('a1')
        allocated['attributes'] = {'state': 'allocated', 'bed_id': 'b1', 'station_id': 's1'}
        waiting = app('w1')
        report = analyze([allocated, waiting], [bed('b1')], [])
        self.assertEqual(entry(report, 'a1')['classification'], 'allocated')
        row = entry(report, 'w1')
        self.assertEqual(row['classification'], 'waitlisted')
        self.assertEqual(row['nights'][0]['blocking_reservation_ids'], ['committed:a1'])
        # 试算不会把已承诺的床夜再给相交申请
        self.assertEqual(report['simulation']['still_waiting'], ['w1'])
        self.assertEqual(report['simulation']['integrity']['double_promises'], [])

    def test_existing_reservation_takes_precedence_over_committed_marker(self):
        allocated = app('a1')
        allocated['attributes'] = {'state': 'allocated', 'bed_id': 'b1', 'station_id': 's1'}
        reservations = [res('r9', 'b1', '2026-11-20', '2026-11-22')]
        report = analyze([allocated], [bed('b1')], reservations)
        target = cell(report, 's1', '2026-11-20')
        self.assertEqual(target['confirmed'], 1)
        self.assertEqual(target['confirmed_reservation_ids'], ['r9'])


class SideEffectFreeTests(unittest.TestCase):
    def test_inputs_are_not_mutated(self):
        beds = [bed('b1')]
        reservations = [res('r1', 'b1', '2026-11-20', '2026-11-22', status='held')]
        applications = [app('w1')]
        snapshot = copy.deepcopy((applications, beds, reservations))
        analyze(applications, beds, reservations, scenario=Scenario('释放r1', release_reservation_ids=('r1',)))
        self.assertEqual((applications, beds, reservations), snapshot)

    def test_service_report_does_not_touch_database(self):
        with tempfile.TemporaryDirectory() as folder:
            database = Database(str(Path(folder) / 'test.sqlite3'))
            service = ShelterService(database, today=lambda: TODAY)
            before_counts, before_pending = database.counts(), len(database.pending_messages())
            result = service.pressure_report([app('w1')], [bed('b1')], [res('r1', 'b1', '2026-11-20', '2026-11-22')], dt.date(2026, 11, 19), dt.date(2026, 11, 24),
                                             scenarios=[{'name': '加床', 'capacity_adjustments': {'s1': 1}}])
            self.assertEqual(database.counts(), before_counts)
            self.assertEqual(len(database.pending_messages()), before_pending)
            self.assertIn('base', result)
            self.assertEqual(result['comparison']['scenarios'][0]['resolved_application_ids'], ['w1'])
            json.dumps(result)  # 报告必须可直接序列化


class CliTests(unittest.TestCase):
    def test_pressure_command(self):
        from shelter_exchange.cli import main
        fixture = {
            'context': {'beds': [bed('b1')], 'reservations': [res('r1', 'b1', '2026-11-20', '2026-11-22', status='held')]},
            'horizon': {'start': '2026-11-19', 'end': '2026-11-24'},
            'applications': [app('w1')],
            'scenarios': [{'name': '释放r1', 'release_reservation_ids': ['r1']}],
        }
        with tempfile.TemporaryDirectory() as folder:
            fixture_path = Path(folder) / 'fixture.json'
            output_path = Path(folder) / 'out.json'
            fixture_path.write_text(json.dumps(fixture), encoding='utf-8')
            code = main(['--database', str(Path(folder) / 'db.sqlite3'), 'pressure',
                         '--fixture', str(fixture_path), '--output', str(output_path)])
            self.assertEqual(code, 0)
            payload = json.loads(output_path.read_text(encoding='utf-8'))
            pressure = payload['pressure']
            self.assertEqual(pressure['base']['summary']['by_classification']['waitlisted'], 1)
            self.assertEqual(pressure['comparison']['scenarios'][0]['resolved_application_ids'], ['w1'])


if __name__ == '__main__':
    unittest.main()
