{{ fullname | escape | underline}}

.. automodule:: {{ fullname }}

   {% block modules %}
   {% if modules %}
   Submodules
   ----------
   .. autosummary::
      :toctree:
      :recursive:
   {% for item in modules %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   .. only:: never

         nothing

   {% block classes %}
   {% if classes %}
   Classes
   -------
   .. autosummary::
      :nosignatures:
   {% for item in classes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block functions %}
   {% if functions %}
   Functions
   ---------
   .. autosummary::
      :nosignatures:
   {% for item in functions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block descriptions %}
   {% if functions or classes %}
   Descriptions
   ------------
   {% endif %}
   {% endblock %}
