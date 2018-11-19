VERSION = 15

# PSMC average power for each state (fep_count, vid_board, clocking)
# [fep_count, vid_board, clocking, power_avg]
psmc_power = ((0, 0, 0, 43.0),
              (1, 0, 0, 27.0),
              (2, 0, 0, 42.0),
              (3, 0, 0, 68.0),
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
# Note that the quantile needs to be in the set (1, 5, 16, 50, 84, 95, 99)
validation_limits = { 'DP_PITCH' : ((1, 5.0),
                                    (99, 5.0),
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
validation_scale_count = { 'OBSID': 2,
                           'HETG': 2,
                           'LETG': 2,
                           'PCAD_MODE': 2,
                           'DITHER': 2}

bad_times = [{'start': '2015:006:08:22:59.000',
              'stop': '2015:009:00:00:00.000'},
             {'start': '2015:012:00:30:00.000',
              'stop': '2015:013:05:26:54.011'},
             {'start': '2015:264:00:00:00.000',
              'stop': '2015:267:00:00:00.000'},
             {'start': '2016:063:12:00:00.000',
              'stop': '2016:065:06:00:00.000'},
             {'start': '2016:234:07:24:00.000',
              'stop': '2016:235:05:18:00.000'},
             {'start': '2016:324:12:59:32.000',
              'stop': '2016:326:06:00:00.000'},
             {'start': '2016:344:07:35:00.000',
              'stop': '2016:345:05:38:00.000'},
             {'start': '2017:066:00:00:00.000',
              'stop': '2017:067:08:00:00.000'},
             {'start': '2017:068:16:00:00.000',
              'stop': '2017:070:05:00:00.000'},
             {'start': '2017:090:18:00:00.000',
              'stop': '2017:092:04:00:00.000'},
             {'start': '2018:079:13:15:00.000',
              'stop': '2018:079:13:30:00.000'},
             {'start': '2018:283:13:54:00.000',
              'stop': '2018:286:12:30:00.000'},
             {'start': '2018:292:23:55:00.000',
              'stop': '2018:293:07:10:00.000'},
             ]


if __name__ == '__main__':
    print VERSION
