# SMS-lightcurve

## Description
The purpose of this code is to compute semi-analytic light curves of the explosion of supermassive stars.

## Publication
This code is associated to the publication
'Signatures of Exploding Supermassive PopIII Stars at High Redshift in JWST, EUCLID and Roman Space Telescope' Cédric Jockel et al. 2025,
and is used to run the model descibed in the paper.


## Usage
To execute the code, simply run
    
    python3 paper_figures.py

No external dependencies are needed apart from standard Python packages. The **code may run for more than an hour** if executed like this, becasue it will run all scripts to produce the figures in the paper back-to-back. If you wish to only reproduce individual figures, you may look at the bottom of paper_figures.py in the __main__ function and (un)-comment the corresponding lines of code.


## License
This sowtware is open-source and may be used, downloaded, modified and redistributed according to the GNU General Public License v3.0. Any derivative works must credit the original authors and cite the above-mentioned publication.

