========================================================================== 
Convolutional Neural Networks for analyzing atomic-resolution HRTEM images
==========================================================================


This is a collection of Python modules, scripts and Jupyter notebooks
for using Convolutional Neural Networks to identify atoms, atomic
columns and possibly other atomic-scale structures in High Resolution
Transmission Electron Microscopy (HRTEM) images.

This code is based on Jacob Madsen's code at
https://github.com/jacobjma/atomic-resolution-tensorflow.  Major parts
of the code are Jacob's, but the neural network itself has been
reimplemented with `Keras <http://keras.io>`_ instead of TensorFlow,
since the latter requires a significant amount of machine learning
expertise (or at least did when Jacob wrote his code).  Keras provides
a high-level interface and sensible defaults for most parameters.

The neural network can be trained on simulated images and subsequently
used on "real" HRTEM images (and image sequences) - this is
particularly useful for analyzing large amounts of images and long sequences.

What can it do?
===============

We have trained networks on Gold nanoparticles and used the network to
track surface diffusion of gold nanoparticles on an oxide support
under the influence of small amounts of gasses.

We have also trained a network on graphene-like sheets, and used the
network to find atoms in defected sheets of graphene, and thus
calculate the strain field in the graphene.

The methods are described in [JAMAD1]_


Microscopy & Microanalysis 2018
===============================

This work will be presented at the `M&M 2018 conference
<https://www.microscopy.org/mandm/2018/>`_

A snapshot of the code at the time of presentation, as well as Jupyter
Notebooks creating many of the figures for the slides are available in
a folder in this project.

A PDF of my presentation will appear soon here.

Installation
============

Installation instructions will appear here (Real Soon Now).

Folders and Files
=================

To be written ...



References
==========

.. [JAMAD1] Jacob Madsen, Pei Liu, Jens Kling, Jakob Birkedal Wagner,
	    Thomas Willum Hansen, Ole Winther and Jakob Schiøtz: *A
	    Deep Learning Approach to Identify Local Structures in
	    Atomic‐Resolution Transmission Electron Microscopy
	    Images*, Adv. Theory Simul. (in press, 2018).  DOI:
	    `10.1002/adts.201800037
	    <https://doi.org/10.1002/adts.201800037>`_.  Preprint
	    available at https://arxiv.org/abs/1802.03008
	    
