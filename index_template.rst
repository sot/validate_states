======================================
Commanded States Database Verification
======================================
.. role:: red

{% if proc.errors %}
Processing Errors
-----------------
.. class:: red
{% endif %}

Summary
--------         
.. class:: borderless

====================  =============================================
{% if opt.loaddir %}
Load directory        {{opt.loaddir}}
{% endif %}
Run time              {{proc.run_time}} by {{proc.run_user}}
Run log               `<run.dat>`_
States                `<states.dat>`_
====================  =============================================

===============
MSID Validation
===============

MSID quantiles
---------------

.. csv-table:: 
   :header: "MSID", "1%", "5%", "16%", "50%", "84%", "95%", "99%"
   :widths: 15, 10, 10, 10, 10, 10, 10, 10

{% for plot in plots_validation %}
{% if plot.quant01 %}
   {{plot.msid}},{{plot.quant01}},{{plot.quant05}},{{plot.quant16}},{{plot.quant50}},{{plot.quant84}},{{plot.quant95}},{{plot.quant99}}
{% endif %}
{% endfor%}


{% if valid_viols %}
Validation Violations
---------------------

.. csv-table:: 
   :header: "MSID", "Quantile", "Value", "Limit"
   :widths: 15, 10, 10, 10

{% for viol in valid_viols %}
   {{viol.msid}},{{viol.quant}},{{viol.value}},{{viol.limit|floatformat:2}}
{% endfor%}

{% else %}
No Validation Violations
{% endif %}    


{% for plot in plots_validation %}
{{ plot.msid }}
-----------------------
{% if plot.diff_only %}
Black = difference values
{% else %}
Red = telemetry, blue = model
{% endif %}

.. image:: {{plot.lines}}

{% if plot.histlog %}
.. image:: {{plot.histlog}}
{{ plot.diff_count }} non-identical samples of {{ plot.samples }} samples.
{% else %}
{{ plot.diff_count }} non-identical samples of {{ plot.samples }} samples.
No histogram provided.
{% endif %}

{% if plot.histlin %}
.. image:: {{plot.histlin}}
{% endif %}

{% endfor %}
