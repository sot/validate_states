# Set the task name
TASK = validate_states

# Versions
VER_TOOL = `python validate_states.py --version`
VER_CAL = `python -m characteristics`
VER_MINOR = 0

# Uncomment the correct choice indicating either SKA or TST flight environment
FLIGHT_ENV = SKA

SHARE = validate_states.py characteristics.py VERSION run_daily_state_check.py
DATA = index_template.rst validate_states.css VERSION task_schedule.cfg

include /proj/sot/ska/include/Makefile.FLIGHT

.PHONY: dist install docs version

# Make a versioned distribution.  Could also use an EXCLUDE_MANIFEST
dist: version
	mkdir $(TASK)-$(VER)
	tar --exclude CVS --exclude "*~" --create --files-from=MANIFEST --file - \
	 | (tar --extract --directory $(TASK)-$(VER) --file - )
	tar --create --verbose --gzip --file $(TASK)-$(VER).tar.gz $(TASK)-$(VER)
	rm -rf $(TASK)-$(VER)

version:
	echo "$(VER_TOOL).$(VER_CAL).$(VER_MINOR)" > VERSION

docs:
	cd docs ; \
	make html

install: version
#  Uncomment the lines which apply for this task
	mkdir -p $(INSTALL_SHARE)
	mkdir -p $(INSTALL_DATA)

	rsync --times --cvs-exclude $(SHARE) $(INSTALL_SHARE)/
	rsync --times --cvs-exclude $(DATA) $(INSTALL_DATA)/

