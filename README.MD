# Faint Solar System Object Detection Service

[![Template](https://img.shields.io/badge/Template-LINCC%20Frameworks%20Python%20Project%20Template-brightgreen)](https://lincc-ppt.readthedocs.io/en/latest/)

## Project Overview

This repository contains a set of Python modules designed to identify potential low signal-to-noise (S/N) detections of solar system objects specified by users. 
The service searches for suitable survey images along the object's probable path and returns detections (within SNR and distance constraints) to the user, along with 
associated metadata and optionally, image cutouts.

## Features

- Obtain list of solar system objects to query from users via web form or API
- Compute expected positions and uncertainty regions of given objects
- Identify survey exposures containing the object's predicted position
- Generate cutout images of each object's uncertainty region
- Retrieve existing higher-S/N (S/N>=5) survey sources in each uncertainty region
- Detect new low-S/N (S/N<5) sources within the uncertainty region
- Conduct forced photometry at the expected position of given objects

<h1 align="center">
<img src="https://raw.githubusercontent.com/lsst-sssc/ssoforcedphot/ephemeris_service/docs/SSSC_Faint_object_service.png" width="500">
</h1><br>

## Installation

(Instructions for installation will be added once more components are completed)

## License

(Add appropriate license information)

## Contact

(Add contact information or link to issues page)

