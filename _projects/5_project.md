---
layout: page
title: Clustering Value Systems and Worldviews
description: Unsupervised learning using different clustering and dimensionality reduction methods to analyse the results of a questionnaire about values and worldviews.
img: assets/img/clustering.png
importance: 3
category: individual
---


# Context

* Many sociological theories suggest that people can be divided into distinct groups based on their values and beliefs about the world. 
* The theories disagree on the number and characteristics of the subsets.
* Clustering can be a good way of finding an appropriate number of subsets of people and understanding the properties of these subsets.
* Aim: to find the subsets of people who share the same values and beliefs to improve policy making.

## Dataset Reference

* **source**: https://www.worldvaluessurvey.org/WVSDocumentationWV7.jsp download "WVS Cross-National Wave 7 csv v6 0.zip"
* **authors**: Haerpfer, C., Inglehart, R., Moreno, A., Welzel, C., Kizilova, K., Diez-Medrano J., M. Lagos, P. Norris, E. Ponarin & B. Puranen
* **license**: 
    * "These data files are available without restrictions, provided: <br>
    a) that they are used for non-profit purposes; <br>
    b) correct citations are provided and sent to the World Values Survey Association for each publication of results based in part or entirely on these data files; <br>
    c) the data files themselves are not redistributed; <br>
    d) proper citation to the WVS data is included into the references list of the publication (citation format available for downloading in the Documentation section)."
* **purpose**: "The WVS seeks to help scientists and policy makers understand changes in the beliefs, values and motivations of people throughout the world."

## Dataset Suitability

* The survey is designed to assess the social, political, economic, religious and cultural values of people around the world.
* Claims to be the largest non-commercial academic social survey programme.
* Converts values into scales and includes the results of the derived values. 


# The Project
> Note: Below is only part of the implementation. The whole code with all classes and functions can be found in [this repository](https://github.com/lartmann/Clustering-Value-Systems-and-Worldviews).

{::nomarkdown}
<div style="background-color: white; color: black; padding: 1em; border-radius: 8px;">
{% assign jupyter_path = "assets/jupyter/analysis.ipynb" | relative_url %}
{% capture notebook_exists %}{% file_exists assets/jupyter/analysis.ipynb %}{% endcapture %}
{% if notebook_exists == "true" %}
{% jupyter_notebook jupyter_path %}
{% else %}

<p>Sorry, the notebook you are looking for does not exist.</p>
{% endif %}
</div>
{:/nomarkdown}