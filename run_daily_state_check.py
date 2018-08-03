#!/usr/bin/env python

from __future__ import print_function

"""
========================
run_daily_state_check
========================

This code calls validate_states for daily trending.

"""

import os
import re
from Ska.File import chdir
from Chandra.Time import DateTime

TASK = 'validate_states'

ROOT = os.environ.get('VALIDATE_STATES') or os.environ['SKA']
CHECK_EXE = os.path.join(ROOT, 'share', TASK, "{}.py".format(TASK))
TASK_DATA = os.path.join(ROOT, 'data', TASK, 'daily')


def get_options():
    from optparse import OptionParser
    parser = OptionParser()
    parser.set_defaults()
    parser.add_option("--run_start_time",
                      default=DateTime().date,
                      help="Reference start time for end of telemetry range")
    parser.add_option("--telem_days",
                      type='float',
                      default=21.0,
                      help="Number of days of telemetry to retrieve")
    parser.add_option("--data_dir",
                      default=TASK_DATA,
                      help="parent directory for data by year/day")
    opt, args = parser.parse_args()
    return opt, args


def main(opt):
    run_time_date = DateTime(opt.run_start_time).date
    date_match = re.search("^(\d{4}):(\d{3}):", run_time_date)
    year = date_match.group(1)
    day = date_match.group(2)
    day_dir = os.path.join(opt.data_dir, year, day)

    if not os.path.exists(day_dir):
        os.makedirs(day_dir)
    with chdir(opt.data_dir):
        local_day_dir = os.path.join(year, day)
        if os.path.exists("current"):
            os.unlink("current")
        os.system("ln -s {} current".format(local_day_dir))

    print(CHECK_EXE)

    # Kadi
    os.system(
        "%s --run_start_time %s --days %s --outdir %s --dbi kadi"
        % (CHECK_EXE, run_time_date, opt.telem_days, day_dir))

    # Sqlite
    sqlite_day_dir = os.path.join(day_dir, 'sqlite')
    sqlite_db = os.path.join(os.environ['SKA'], 'data', 'cmd_states', 'cmd_states.db3')
    os.system(
        "%s --run_start_time %s --days %s --outdir %s --dbi sqlite --server %s"
        % (CHECK_EXE, run_time_date, opt.telem_days, sqlite_day_dir, sqlite_db))

    # Sybase
    sybase_day_dir = os.path.join(day_dir, 'sybase')
    os.system("%s --run_start_time %s --days %s --outdir %s"
              % (CHECK_EXE, run_time_date, opt.telem_days, sybase_day_dir))

if __name__ == '__main__':

    opt, args = get_options()
    main(opt)
