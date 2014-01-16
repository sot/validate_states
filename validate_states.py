#!/usr/bin/env python

"""
===============
validate_states
===============

Compare cmd_states to telemetry values to verify that as-run commands
match predictions.
Based on psmc_check.py validation
"""

import sys
import os
import logging
import re
from pprint import pformat
import time
import shutil
import pickle
import numpy as np
from itertools import count
import django.template
import django.conf
import docutils.writers.html4css1

# Matplotlib setup
# Use Agg backend for command-line (non-interactive) operation
import matplotlib
if __name__ == '__main__':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

import Ska.Matplotlib
from Ska.Matplotlib import plot_cxctime
import Ska.DBI
import Ska.Shell
#import Ska.Table
import Ska.Numpy
import Ska.engarchive.fetch_sci as fetch
from Chandra.Time import DateTime
from mica.quaternion import Quat, normalize

import Chandra.cmd_states as cmd_states
import characteristics
from characteristics import validation_limits, validation_count

TASK = 'validate_states'
VERSION = 1
TASK_DATA = os.path.join(os.environ['SKA'], 'data', TASK)
URL = "http://cxc.harvard.edu/mta/ASPECT/" + TASK
logger = logging.getLogger(TASK)

LABELS = {'dp_pitch': 'Pitch (degrees)',
          'obsid': 'OBSID',
          'tscpos': 'SIM-Z (steps/1000)',
          'pcad_mode': 'PCAD MODE',
          'dither': 'Dither Enabled',
          'letg': 'LETG',
          'hetg': 'HETG',
          'power': 'ACIS power (watts)',
          'pointing': 'Radial Offset (arcsec)',
          'roll': 'Roll Offset (arcsec)'}


SCALES = {'tscpos': 1000.,
          'dither': 1.}

FMTS = {'dp_pitch': '%.3f',
        'obsid': '%d',
        'dither': '%d',
        'hetg': '%d',
        'letg': '%d',
        'pcad_mode': '%d',
        'power': '%.2f',
        'tscpos': '%d',
        'pointing': '%.2f',
        'roll': '%.2f',
        }

MODE_SOURCE = {'pcad_mode': 'aopcadmd',
               'dither': 'aodithen'}

MODE_MSIDS = {'pcad_mode': ['NMAN', 'NPNT', 'NSUN', 'NULL',
                            'PWRF', 'RMAN', 'STBY'],
              'dither': ['ENAB', 'DISA'],
              'hetg': ['INSE', 'RETR'],
              'letg': ['INSE', 'RETR'],
              }


def get_options():
    from optparse import OptionParser
    parser = OptionParser()
    parser.set_defaults()
    parser.add_option("--outdir",
                      default="out",
                      help="Output directory")
    parser.add_option("--days",
                      type='float',
                      default=21.0,
                      help="Days of validation data (days)")
    parser.add_option("--run_start_time",
                      help="Mock tool run time for regression testing")
    parser.add_option("--traceback",
                      default=True,
                      help='Enable tracebacks')
    parser.add_option("--verbose",
                      type='int',
                      default=1,
                      help="Verbosity (0=quiet, 1=normal, 2=debug)")
    parser.add_option("--version",
                      action='store_true',
                      help="Print version")

    opt, args = parser.parse_args()
    return opt, args


def config_logging(outdir, verbose):
    """Set up file and console logger.
    See http://docs.python.org/library/logging.html#logging-to-multiple-destinations
    """
    # Disable auto-configuration of root logger by adding a null handler.
    # This prevents other modules (e.g. Chandra.cmd_states) from generating
    # a streamhandler by just calling logging.info(..).
    class NullHandler(logging.Handler):
        def emit(self, record):
            pass
    rootlogger = logging.getLogger()
    rootlogger.addHandler(NullHandler())

    loglevel = {0: logging.CRITICAL,
                1: logging.INFO,
                2: logging.DEBUG}.get(verbose, logging.INFO)

    logger = logging.getLogger(TASK)
    logger.setLevel(loglevel)

    formatter = logging.Formatter('%(message)s')

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    filehandler = logging.FileHandler(
        filename=os.path.join(outdir, 'run.dat'), mode='w')
    filehandler.setFormatter(formatter)
    logger.addHandler(filehandler)


def get_telem_values(tstart, msids, days=14, dt=32.8, name_map={}):
    """
    Fetch last ``days`` of available ``msids`` telemetry values before
    time ``tstart``.

    :param tstart: start time for telemetry (secs)
    :param msids: fetch msids list
    :param days: length of telemetry request before ``tstart``
    :param dt: sample time (secs)
    :param name_map: dict mapping msid to recarray col name
    :returns: np recarray of requested telemetry values from fetch
    """
    tstart = DateTime(tstart).secs
    start = DateTime(tstart - days * 86400).date
    stop = DateTime(tstart).date
    logger.info('Fetching telemetry between %s and %s' % (start, stop))
    msidset = fetch.Msidset(msids, start, stop)
    start = max(x.times[0] for x in msidset.values())
    stop = min(x.times[-1] for x in msidset.values())
    msidset.interpolate(dt, start, stop)

    # Finished when we found at least 10 good records (5 mins)
    if len(msidset.times) < 10:
        raise ValueError('Found no telemetry within %d days of %s'
                         % (days, str(tstart)))

    outnames = ['date'] + [name_map.get(x, x) for x in msids]
    out = np.rec.fromarrays([msidset.times] +
                            [msidset[x].vals for x in msids],
                            names=outnames)
    return out


def get_power(states):
    """
    Determine the power value in each state by finding the entry in calibration
    power table with the same ``fep_count``, ``vid_board``, and ``clocking``
    values.

    :param states: input states
    :rtype: numpy array of power corresponding to states
    """

    # Make a tuple of values that define a unique power state
    powstate = lambda x: tuple(x[col] for col
                               in ('fep_count', 'vid_board', 'clocking'))

    # psmc_power charactestic is a list of 4-tuples (fep_count vid_board
    # clocking power_avg).  Build a dict to allow access to power_avg for
    # available (fep_count vid_board clocking) combos.
    power_states = dict((row[0:3], row[3]) for row
                        in characteristics.psmc_power)
    try:
        powers = [power_states[powstate(x)] for x in states]
    except KeyError:
        raise ValueError('Unknown power state: %s' % str(powstate(x)))

    return powers


def smoothed_power(tlm):
    """Calculate the smoothed PSMC power from telemetry ``tlm``.
    """
    pwrdea = Ska.Numpy.smooth(tlm['1de28avo'] * tlm['1deicacu'],
                              window_len=33, window='flat')
    pwrdpa = Ska.Numpy.smooth(tlm['1dp28avo'] * tlm['1dpicacu']
                              + tlm['1dp28bvo'] * tlm['1dpicbcu'],
                              window_len=21, window='flat')

    return pwrdea + pwrdpa


def main(opt):
    opt, args = get_options()
    if not os.path.exists(opt.outdir):
        os.mkdir(opt.outdir)

    config_logging(opt.outdir, opt.verbose)

    # Store info relevant to processing for use in outputs
    proc = dict(run_user=os.environ['USER'],
                run_time=time.ctime(),
                errors=[],
                )
    logger.info('#####################################################################')
    logger.info('# %s run at %s by %s' % (os.path.dirname(__file__),
                                          proc['run_time'], proc['run_user']))
    logger.info('# version = %s' % VERSION)
    logger.info('# characteristics version = %s' % characteristics.VERSION)
    logger.info('#####################################################################\n')

    logger.info('Command line options:\n%s\n' % pformat(opt.__dict__))

    # Connect to database (NEED TO USE aca_read)
    tnow = DateTime(opt.run_start_time).secs
    tstart = tnow

    # Get temperature telemetry for 3 weeks prior to min(tstart, NOW)
    tlm = get_telem_values(tstart,
                           ['sim_z', 'dp_pitch',
                            'aodithen', 'cobsrqid',
                            'aopcadmd', '4ootgsel', '4ootgmtn',
                            'aoattqt1', 'aoattqt2',
                            'aoattqt3', 'aoattqt4',
                            '1de28avo', '1deicacu',
                            '1dp28avo', '1dpicacu',
                            '1dp28bvo', '1dpicbcu'],
                           days=opt.days,
                           name_map={'sim_z': 'tscpos',
                                     'cobsrqid': 'obsid'})

    tlm['tscpos'] = tlm['tscpos'] * -397.7225924607
    outdir = opt.outdir
    states = get_states(tlm[0].date, tlm[-1].date)
    write_states(opt, states)
    tlm = Ska.Numpy.add_column(tlm, 'power', smoothed_power(tlm))

    # Interpolate states onto the tlm.date grid
    state_vals = cmd_states.interpolate_states(states, tlm.date)

    # Calculate angle/roll differences in cmd vs tlm quaternions
    raw_tlm_q = np.vstack([tlm['aoattqt' + str(n)] for n
                           in [1, 2, 3, 4]]).transpose()
    raw_state_q = np.vstack([state_vals[n] for n
                             in ['q1', 'q2', 'q3', 'q4']]).transpose()
    tlm_q = normalize(raw_tlm_q)
    # only use values that aren't NaNs
    good = np.isnan(np.sum(tlm_q, axis=-1)) == False
    # only do the pointing diffs in NPNT
    # the maneuver interpolation is just too messy
    npnt = state_vals['pcad_mode'] == 'NPNT'
    state_q = normalize(raw_state_q)
    dot_q = np.sum(tlm_q[good & npnt] * state_q[good & npnt], axis=-1)
    angle_diff = np.degrees(2 * np.arccos(dot_q))
    angle_diff = np.min([angle_diff, 360 - angle_diff], axis=0)
    roll_diff = Quat(tlm_q[good & npnt]).roll - Quat(state_q[good & npnt]).roll
    roll_diff = np.min([roll_diff, 360 - roll_diff], axis=0)

    for msid in MODE_SOURCE:
        tlm_col = np.zeros(len(tlm))
        state_col = np.zeros(len(tlm))
        for mode, idx in zip(MODE_MSIDS[msid], count()):
            tlm_col[tlm[MODE_SOURCE[msid]] == mode] = idx
            state_col[state_vals[msid] == mode] = idx
        tlm = Ska.Numpy.add_column(tlm, msid, tlm_col)
        state_vals = Ska.Numpy.add_column(state_vals,
                                          "{}_pred".format(msid), state_col)

    for msid in ['letg', 'hetg']:
        txt = np.repeat('RETR', len(tlm))
        # use a combination of the select telemetry and the insertion telem to
        # approximate the state_vals values
        txt[(tlm['4ootgsel'] == msid.upper())
            & (tlm['4ootgmtn'] == 'INSE')] = 'INSE'
        tlm_col = np.zeros(len(tlm))
        state_col = np.zeros(len(tlm))
        for mode, idx in zip(MODE_MSIDS[msid], count()):
            tlm_col[txt == mode] = idx
            state_col[state_vals[msid] == mode] = idx
        tlm = Ska.Numpy.add_column(tlm, msid, tlm_col)
        state_vals = Ska.Numpy.add_column(state_vals,
                                          "{}_pred".format(msid), state_col)

    diff_only = {'pointing': {'diff': angle_diff * 3600,
                              'date': tlm.date[ok]},
                 'roll': {'diff': roll_diff * 3600,
                          'date': tlm.date[ok]}}

    pred = {'dp_pitch': state_vals.pitch,
            'obsid': state_vals.obsid,
            'dither': state_vals['dither_pred'],
            'pcad_mode': state_vals['pcad_mode_pred'],
            'letg': state_vals['letg_pred'],
            'hetg': state_vals['hetg_pred'],
            'tscpos': state_vals.simpos,
            'power': state_vals.power,
            'pointing': 1,
            'roll': 1}

    plots_validation = []
    logger.info('Making validation plots and quantile table')
    quantiles = (1, 5, 16, 50, 84, 95, 99)
    # store lines of quantile table in a string and write out later
    quant_table = ''
    quant_head = ",".join(['MSID'] + ["quant%d" % x for x in quantiles])
    quant_table += quant_head + "\n"
    for fig_id, msid in enumerate(sorted(pred)):
    #for fig_id, msid in enumerate(['obsid']):
        plot = dict(msid=msid.upper())
        fig = plt.figure(10 + fig_id, figsize=(7, 3.5))
        fig.clf()
        scale = SCALES.get(msid, 1.0)
        if msid not in diff_only:
            if msid in MODE_MSIDS:
                state_msid = np.zeros(len(tlm))
                for mode, idx in zip(MODE_MSIDS[msid], count()):
                    state_msid[state_vals[msid] == mode] = idx
                ticklocs, fig, ax = plot_cxctime(tlm.date,
                                                 tlm[msid], fig=fig, fmt='-r')
                ticklocs, fig, ax = plot_cxctime(tlm.date,
                                                 state_msid, fig=fig, fmt='-b')
                plt.yticks(range(len(MODE_MSIDS[msid])), MODE_MSIDS[msid])
            else:
                ticklocs, fig, ax = plot_cxctime(tlm.date,
                                                 tlm[msid] / scale, fig=fig, fmt='-r')
                ticklocs, fig, ax = plot_cxctime(tlm.date,
                                                 pred[msid] / scale, fig=fig, fmt='-b')
        else:
            ticklocs, fig, ax = plot_cxctime(diff_only[msid]['date'],
                                             diff_only[msid]['diff'] / scale, fig=fig, fmt='-k')
        plot['diff_only'] = msid in diff_only
        ax.set_title(msid.upper() + ' validation')
        ax.set_ylabel(LABELS[msid])
        filename = msid + '_valid.png'
        outfile = os.path.join(outdir, filename)
        logger.info('Writing plot file %s' % outfile)
        fig.savefig(outfile)
        plot['lines'] = filename

        # Make quantiles
        if msid not in diff_only:
            diff = np.sort(tlm[msid] - pred[msid])
        else:
            diff = np.sort(diff_only[msid]['diff'])
        quant_line = "%s" % msid
        for quant in quantiles:
            quant_val = diff[(len(diff) * quant) // 100]
            plot['quant%02d' % quant] = FMTS[msid] % quant_val
            quant_line += (',' + FMTS[msid] % quant_val)
        quant_table += quant_line + "\n"

        plot['samples'] = len(diff)
        plot['diff_count'] = len(np.flatnonzero(diff))
        # if there are only a few residuals, don't bother with histograms
        if ((msid.upper() in validation_count)
                and (plot['diff_count'] < validation_count[msid.upper()])):
            plots_validation.append(plot)
            continue

        for histscale in ('lin', 'log'):
            fig = plt.figure(20 + fig_id, figsize=(4, 3))
            fig.clf()
            ax = fig.gca()
            ax.hist(diff / scale, bins=50, log=(histscale == 'log'))
            ax.set_title(msid.upper() + ' residuals: data - model')
            ax.set_xlabel(LABELS[msid])
            fig.subplots_adjust(bottom=0.18)
            filename = '%s_valid_hist_%s.png' % (msid, histscale)
            outfile = os.path.join(outdir, filename)
            logger.info('Writing plot file %s' % outfile)
            fig.savefig(outfile)
            plot['hist' + histscale] = filename

        plots_validation.append(plot)

    filename = os.path.join(outdir, 'validation_quant.csv')
    logger.info('Writing quantile table %s' % filename)
    f = open(filename, 'w')
    f.write(quant_table)
    f.close()

    # If run_start_time is specified this is likely for regression testing
    # or other debugging.  In this case write out the full predicted and
    # telemetered dataset as a pickle.
    if opt.run_start_time:
        filename = os.path.join(outdir, 'validation_data.pkl')
        logger.info('Writing validation data %s' % filename)
        f = open(filename, 'w')
        pickle.dump({'pred': pred, 'tlm': tlm}, f, protocol=-1)
        f.close()

    valid_viols = make_validation_viols(plots_validation)
    if len(valid_viols) > 0:
        # generate daily plot url if outdir in expected year/day format
        daymatch = re.match('.*(\d{4})/(\d{3})', opt.outdir)
        if daymatch:
            url = os.path.join(URL, daymatch.group(1), daymatch.group(2))
            logger.info('validation warning(s) at %s' % url)
        else:
            logger.info('validation warning(s) in output at %s' % opt.outdir)

    write_index_rst(opt, proc, plots_validation, valid_viols)
    rst_to_html(opt, proc)


def rst_to_html(opt, proc):
    """Run rst2html.py to render index.rst as HTML"""

    # First copy CSS files to outdir
    dirname = os.path.dirname(docutils.writers.html4css1.__file__)
    shutil.copy2(os.path.join(dirname, 'html4css1.css'), opt.outdir)

    shutil.copy2(os.path.join(TASK_DATA, 'validate_states.css'), opt.outdir)

    spawn = Ska.Shell.Spawn(stdout=None)
    infile = os.path.join(opt.outdir, 'index.rst')
    outfile = os.path.join(opt.outdir, 'index.html')
    status = spawn.run(['rst2html.py',
                        '--stylesheet-path=%s' % os.path.join(opt.outdir, 'validate_states.css'),
                        infile, outfile])
    if status != 0:
        proc['errors'].append('rst2html.py failed with status %d: check run log.' % status)
        logger.error('rst2html.py failed')
        logger.error(''.join(spawn.outlines) + '\n')

    # Remove the stupid <colgroup> field that docbook inserts.  This
    # <colgroup> prevents HTML table auto-sizing.
    del_colgroup = re.compile(r'<colgroup>.*?</colgroup>', re.DOTALL)
    outtext = del_colgroup.sub('', open(outfile).read())
    open(outfile, 'w').write(outtext)


def write_states(opt, states):
    """Write states recarray to file states.dat"""
    outfile = os.path.join(opt.outdir, 'states.dat')
    logger.info('Writing states to %s' % outfile)
    out = open(outfile, 'w')
    fmt = {'power': '%.1f',
           'pitch': '%.2f',
           'tstart': '%.2f',
           'tstop': '%.2f',
           }
    newcols = list(states.dtype.names)
    newstates = np.rec.fromarrays([states[x] for x in newcols], names=newcols)
    Ska.Numpy.pprint(newstates, fmt, out)
    out.close()


def write_index_rst(opt, proc, plots_validation, valid_viols=None):
    """
    Make output text (in ReST format) in opt.outdir.
    """
    # Django setup (used for template rendering)
    try:
        django.conf.settings.configure()
    except RuntimeError, msg:
        print msg

    outfile = os.path.join(opt.outdir, 'index.rst')
    logger.info('Writing report file %s' % outfile)
    django_context = django.template.Context({'opt': opt,
                                              'valid_viols': valid_viols,
                                              'proc': proc,
                                              'plots_validation': plots_validation,
                                              })
    index_template_file = 'index_template.rst'
    index_template = open(os.path.join(TASK_DATA, index_template_file)).read()
    index_template = re.sub(r' %}\n', ' %}', index_template)
    template = django.template.Template(index_template)
    open(outfile, 'w').write(template.render(django_context))


def get_states(datestart, datestop):
    """Get states exactly covering date range

    :param datestart: start date
    :param datestop: stop date
    :param db: database handle
    :returns: np recarry of states
    """
    logger.info('Connecting to database to get cmd_states')
    db = Ska.DBI.DBI(dbi='sybase', server='sybase',
                     user='aca_read', database='aca')

    datestart = DateTime(datestart).date
    datestop = DateTime(datestop).date
    logger.info('Getting commanded states between %s - %s' %
                 (datestart, datestop))

    # Get all states that intersect specified date range
    cmd = """SELECT * FROM cmd_states
             WHERE datestop > '%s' AND datestart < '%s'
             ORDER BY datestart""" % (datestart, datestop)
    logger.debug('Query command: %s' % cmd)
    states = db.fetchall(cmd)
    db.conn.close()
    logger.info('Found %d commanded states' % len(states))

    # Add power columns to states and tlm
    states = Ska.Numpy.add_column(states, 'power', get_power(states))

    # Set start and end state date/times to match telemetry span.  Extend the
    # state durations by a small amount because of a precision issue converting
    # to date and back to secs.  (The reference tstop could be just over the
    # 0.001 precision of date and thus cause an out-of-bounds error when
    # interpolating state values).
    states[0].tstart = DateTime(datestart).secs - 0.01
    states[0].datestart = DateTime(states[0].tstart).date
    states[-1].tstop = DateTime(datestop).secs + 0.01
    states[-1].datestop = DateTime(states[-1].tstop).date

    return states


def make_validation_viols(plots_validation):
    """
    Find limit violations where MSID quantile values are outside the
    allowed range.
    """

    logger.info('Checking for validation violations')

    viols = []

    for plot in plots_validation:
        # 'plot' is actually a structure with plot info and stats about the
        #  plotted data for a particular MSID.  'msid' can be a real MSID
        #  (1PDEAAT) or pseudo like 'POWER'
        msid = plot['msid']

        # Make sure validation limits exist for this MSID
        if msid not in validation_limits:
            continue

        # Cycle through defined quantiles (e.g. 99 for 99%) and corresponding
        # limit values for this MSID.
        for quantile, limit in validation_limits[msid]:
            # Get the quantile statistic as calculated when making plots
            msid_quantile_value = float(plot['quant%02d' % quantile])

            # Check for a violation and take appropriate action
            if abs(msid_quantile_value) > limit:
                viol = {'msid': msid,
                        'value': msid_quantile_value,
                        'limit': limit,
                        'quant': quantile,
                        }
                viols.append(viol)
                logger.info('WARNING: %s %d%% quantile value of %s exceeds limit of %.2f' %
                            (msid, quantile, msid_quantile_value, limit))

    return viols


if __name__ == '__main__':
    opt, args = get_options()
    if opt.version:
        print VERSION
        sys.exit(0)

    try:
        main(opt)
    except Exception, msg:
        if opt.traceback:
            raise
        else:
            print "ERROR:", msg
            sys.exit(1)
