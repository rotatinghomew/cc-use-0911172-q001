from __future__ import annotations
import argparse
import datetime as dt
import json
from pathlib import Path
from .reporting import Reporter
from .service import ApplicationService
from .storage import Database

def load_json(path:str):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def main(argv=None):
    parser=argparse.ArgumentParser(prog='shelter_exchange')
    parser.add_argument('--database',default=':memory:')
    sub=parser.add_subparsers(dest='command',required=True)
    run=sub.add_parser('run'); run.add_argument('--fixture',required=True); run.add_argument('--output',required=True)
    pressure=sub.add_parser('pressure'); pressure.add_argument('--fixture',required=True); pressure.add_argument('--output',required=True)
    report=sub.add_parser('report'); report.add_argument('--kind')
    args=parser.parse_args(argv); database=Database(args.database)
    if args.command=='run':
        fixture=load_json(args.fixture); service=ApplicationService(database); outputs=[]
        for index,item in enumerate(fixture.get('items',[])):
            outputs.append(service.process(item,fixture.get('context',{}),item.get('request_key',f'item-{index}')))
        payload={'product':'青年人才驿站床位协同与入住履约平台','processed':len(outputs),'outputs':outputs,'report':Reporter(database).overview()}
        Path(args.output).parent.mkdir(parents=True,exist_ok=True); Path(args.output).write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(payload,ensure_ascii=False,sort_keys=True)); return 0
    if args.command=='pressure':
        fixture=load_json(args.fixture); context=fixture.get('context',{}); horizon=fixture['horizon']; service=ApplicationService(database)
        result=service.pressure_report(fixture.get('applications',[]),context.get('beds',[]),context.get('reservations',[]),
            dt.date.fromisoformat(horizon['start']),dt.date.fromisoformat(horizon['end']),
            prior_stays=context.get('prior_stays',()),scenarios=fixture.get('scenarios',()))
        payload={'product':'青年人才驿站床位协同与入住履约平台','pressure':result}
        Path(args.output).parent.mkdir(parents=True,exist_ok=True); Path(args.output).write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(payload,ensure_ascii=False,sort_keys=True)); return 0
    payload=Reporter(database).owner_breakdown(args.kind) if args.kind else Reporter(database).overview(); print(json.dumps(payload,ensure_ascii=False,sort_keys=True)); return 0
