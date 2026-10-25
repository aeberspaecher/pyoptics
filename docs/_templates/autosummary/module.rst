{{ fullname | escape | underline }}

.. automodule:: {{ fullname }}

   {% block classes %}
   {% if classes %}
   .. rubric:: Classes

   .. autosummary::
      :nosignatures:
   {% for item in classes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block functions %}
   {% if functions %}
   .. rubric:: Functions

   .. autosummary::
      :nosignatures:
   {% for item in functions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
