---
layout: page
permalink: /repositories/
title: Projects
description: Curated selected open-source projects I have completed independently, including deep learning and software engineering projects.
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


 <a href="https://github.com/Kazawaryu">
 <img  src="https://bad-apple-github-readme.vercel.app/api?show_bg=1&username=Kazawaryu&show_icons=true&icon_color=CE1D2D&text_color=718096&bg_color=ffffff" />
 </a>


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
