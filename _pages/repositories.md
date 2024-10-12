---
layout: page
permalink: /repositories/
title: Open-Source Research
description: Discover a curated selection of open-source repositories reflecting my contributions to the field of Artificial Intelligence. Explore projects that emphasize collaborative efforts and contribute to the ongoing advancements in this domain.
nav: true
nav_order: 4
---

## GitHub User

<!-- {% if site.data.repositories.github_users %}
<div class="repositories d-flex flex-wrap flex-md-row flex-column justify-content-between align-items-center">
  {% for user in site.data.repositories.github_users %}
    {% include repository/repo_user.html username=user %}
  {% endfor %}
</div> -->


 <img align="right" src="https://bad-apple-github-readme.vercel.app/api?show_bg=1&username=Kazawaryu&show_icons=true&icon_color=CE1D2D&text_color=718096&bg_color=ffffff" />

---

{% if site.repo_trophies.enabled %}
{% for user in site.data.repositories.github_users %}
  {% if site.data.repositories.github_users.size > 1 %}
  <h4>{{ user }}</h4>
  {% endif %}
  <div class="repositories d-flex flex-wrap flex-md-row flex-column justify-content-between align-items-center">
  {% include repository/repo_trophies.html username=user %}
  </div>

  ---

{% endfor %}
{% endif %}
{% endif %}

## GitHub Repositories

{% if site.data.repositories.github_repos %}
<div class="repositories d-flex flex-wrap flex-md-row flex-column justify-content-between align-items-center">
  {% for repo in site.data.repositories.github_repos %}
    {% include repository/repo.html repository=repo %}
  {% endfor %}
</div>
{% endif %}
