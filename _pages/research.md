---
layout: page
permalink: /research/
title: Research
description: Research projects I have led or contributed to.
nav: true
nav_order: 1
---

{% for group in site.data.projects %}
<h2 class="project-section">{{ group.section }}</h2>
{% if group.subtitle %}<p class="project-section-sub">{{ group.subtitle }}</p>{% endif %}

{% for p in group.projects %}
<div class="project-card">
  <div class="row">
    <div class="{% if p.image %}col-sm-8{% else %}col-12{% endif %}">
      <h3 class="project-title">{{ p.title }}</h3>
      <p class="project-meta">
        <i class="fa-solid fa-flask fa-sm"></i> {{ p.org }}
        &nbsp;&middot;&nbsp; <i class="fa-solid fa-calendar fa-sm"></i> {{ p.period }}
        {% if p.sponsor %}&nbsp;&middot;&nbsp; <i class="fa-solid fa-landmark fa-sm"></i> Sponsored by {{ p.sponsor }}{% endif %}
      </p>
      {% if p.award %}<p class="project-award"><i class="fa-solid fa-trophy"></i> {{ p.award }}</p>{% endif %}
      <p class="project-summary">{{ p.summary }}</p>
      {% if p.bullets %}
      <ul class="project-bullets">
        {% for b in p.bullets %}<li>{{ b }}</li>{% endfor %}
      </ul>
      {% endif %}
      <div class="project-tags">
        {% for t in p.tags %}<span class="project-tag">{{ t }}</span>{% endfor %}
      </div>
      {% if p.links %}
      <div class="project-links">
        {% for l in p.links %}
        <a class="btn btn-sm z-depth-0" role="button" href="{% if l.url contains '://' %}{{ l.url }}{% else %}{{ l.url | relative_url }}{% endif %}"><i class="{{ l.icon }}"></i> {{ l.name }}</a>
        {% endfor %}
      </div>
      {% endif %}
    </div>
    {% if p.image %}
    <div class="col-sm-4 project-image">
      {% assign img = p.image | prepend: 'assets/img/' %}
      {% include figure.html path=img class="img-fluid rounded z-depth-1" alt=p.title zoomable=true sizes="(min-width: 576px) 30vw, 95vw" %}
    </div>
    {% endif %}
  </div>
</div>
{% endfor %}
{% endfor %}

<h2 class="project-section">Open Source</h2>
<p class="project-section-sub">More on <a href="https://github.com/{{ site.github_username }}">GitHub</a>.</p>
<ul class="project-bullets">
  <li><a href="https://github.com/Kazawaryu/OptAdDatasets">OptAdDatasets</a> — dataset optimization by complexity, quality, and uncertainty</li>
  <li><a href="https://github.com/Kazawaryu/CARLA_ADA">CARLA-ADA</a> — active data acquisition tool for the CARLA simulator</li>
</ul>
