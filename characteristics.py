VERSION = 8

# PSMC average power for each state (fep_count, vid_board, clocking)
# [fep_count, vid_board, clocking, power_avg]
psmc_power = ((0, 0, 0, 15.0),
              (1, 0, 0, 27.0),
              (2, 0, 0, 42.0),
              (3, 0, 0, 55.0),
              (4, 0, 0, 69.0),
              (5, 0, 0, 88.6),
              (6, 0, 0, 96.6),
              (0, 1, 0, 40.0),
              (1, 1, 0, 58.3),
              (2, 1, 0, 69.0),
              (3, 1, 0, 80.0),
              (4, 1, 0, 92.0),
              (5, 1, 0, 112.3),
              (6, 1, 0, 118.0),
              (0, 1, 1, 40.0),
              (1, 1, 1, 57.0),
              (2, 1, 1, 72.0),
              (3, 1, 1, 85.4),
              (4, 1, 1, 99.2),
              (5, 1, 1, 113.9),
              (6, 1, 1, 129.0),
              (0, 0, 1, 40.0),    # These last 7 states only occur
              (1, 0, 1, 57.0),    # for a short time due to 
              (2, 0, 1, 72.0),    # missing stop science at end of
              (3, 0, 1, 85.4),    # previous load.
              (4, 0, 1, 99.2),
              (5, 0, 1, 113.9),
              (6, 0, 1, 129.0))


# validation limits
# 'msid' : (( quantile, absolute max value ))
validation_limits = { 'DP_PITCH' : ((1, 4.0),
                                    (99, 4.0),
                                    (5, 1.5),
                                    (95, 1.5),),
                      'POINTING': ((1, .05),
                                 (99, .05),),
                      'ROLL': ((1, .05),
                               (99, .05),),
                      'POWER':     ((1, 14.0),
                                    (50, 3.0),
                                    (99, 14.0),),
                      'TSCPOS' :   ((1, 2.0),
                                    (99, 2.0),) }

# number of tolerated differences for string / discrete msids
# 'msid' : n differences before violation recorded
# this is scaled by the number of toggles or expected
# changes in the msid
validation_scale_count = { 'OBSID': 1,
                           'HETG': 1,
                           'LETG': 1,
                           'PCAD_MODE': 1,
                           'DITHER': 1}

bad_times = [{'start': '2015:006:08:22:59.000',
              'stop': '2015:009:00:00:00.000'},
             {'start': '2015:012:00:30:00.000',
              'stop': '2015:013:05:26:54.011'},
             {'start': '2015:264:00:00:00.000',
              'stop': '2015:267:00:00:00.000'},
             {'start': '2016:063:12:00:00.000',
              'stop': '2016:065:06:00:00.000'}]

if __name__ == '__main__':
    print VERSION
