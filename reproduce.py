"""Reproduce the state-retail nowcast tables and their estimates."""
from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    tables=commands.add_parser('tables',help='Create tables from saved aggregate results; no estimation')
    from retail_nowcast.tables import TABLES
    tables.add_argument('--table',choices=['all',*TABLES],default='all')
    tables.add_argument('--results',type=Path,default=ROOT/'summaries')
    tables.add_argument('--output',type=Path,default=ROOT/'output/tables')
    estimate=commands.add_parser('estimate',help='Estimate a reported comparison using authorized source files')
    estimate.add_argument('--exercise',choices=['main','replacement','timing','all'],default='all')
    estimate.add_argument('--data-root',type=Path,required=True)
    estimate.add_argument('--output',type=Path,default=ROOT/'output/estimates')
    appendix=commands.add_parser('mrts-lag',help='Estimate the appendix MRTS-lag comparison')
    appendix.add_argument('--data-root',type=Path,required=True)
    appendix.add_argument('--output',type=Path,default=ROOT/'output/mrts_lag_appendix')
    appendix_tables=commands.add_parser('mrts-lag-tables',help='Create the appendix MRTS-lag table')
    appendix_tables.add_argument('--results',type=Path,default=ROOT/'summaries/mrts_lag')
    appendix_tables.add_argument('--output',type=Path,default=ROOT/'output/mrts_lag_tables')
    args=parser.parse_args()
    if args.command=='mrts-lag':
        from retail_nowcast.mrts_lag import MRTSLagComparison
        result=MRTSLagComparison(args.data_root,args.output).run_appendix()
        print(f'Appendix aggregate results saved to {result}')
        return
    if args.command=='mrts-lag-tables':
        from retail_nowcast.mrts_lag import build_tables
        print(build_tables(args.results,args.output))
        return
    if args.command=='tables':
        from retail_nowcast.tables import build
        created=build(args.results,args.output,args.table)
        print(f'Created {len(created)} tables in {args.output} (CSV and LaTeX).')
    else:
        from retail_nowcast.estimation import Estimation
        result=Estimation(args.data_root,args.output,args.exercise).run()
        print(f'Aggregate results saved to {result}. Use --results {result} when building tables.')

if __name__=='__main__':main()
