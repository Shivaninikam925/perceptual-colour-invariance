# Perceptual Invariance in Human Color Judgement

This project studies perceptual invariance in human color judgement under controlled illumination changes, focusing on representation stability rather than task performance.

## Motivation
Human observers are able to judge colors as perceptually similar despite changes in illumination. This ability suggests the existence of stable perceptual representations that discount certain appearance variations. Understanding such invariance is a central question in visual perception and representation learning.

## Approach
We analyze perceptual invariance by measuring the stability of different color representations under controlled illumination perturbations:
- Global brightness scaling
- Warm–cool color temperature shifts
- Contrast scaling

We compare three representations:
1. Raw RGB values
2. CIELAB color space
3. A minimal learned linear embedding

Invariance is operationalized as the average embedding distance between original colors and their illumination-perturbed versions.

## Key Result
Even a simple learned linear embedding exhibits substantially greater stability under illumination changes than CIELAB, indicating that representation choice plays a critical role in perceptual invariance.

A single diagnostic figure summarizing this comparison is shown in:


## Scope and Limitations
This is a small pilot study using controlled color patches and proxy perceptual similarity. The goal is not to model full human color perception, but to establish a clean, extensible framework that can be expanded with psychophysical data, natural images, or neural representations.

## Status
This repository contains a completed pilot experiment designed to be extended into a longer vision and perception study.

## Author
Shivani Nikam
Second year Btech CSE student(India)
Completed a small pilot study examining perceptual ivariance in human colour judgement under illumination changes, focusing on representation stability rather than task accuracy.
