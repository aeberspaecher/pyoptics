# Beispiel-Notebooks

Diese Jupyter-Notebooks zeigen die einzelnen Teile von pyoptics an Beispielen,
deren Physik man gut nachvollziehen kann, und visualisieren sie ausführlich.
Wo es eine analytische Lösung gibt, wird die Numerik damit verglichen. Die
Notebooks sind ausgeführt gespeichert, die Abbildungen sind also auch ohne
eigene Rechnung sichtbar (z.B. direkt auf GitHub).

| Notebook | Module | Inhalt |
|---|---|---|
| [01_propagation](01_propagation.ipynb) | `propagators`, `analytical`, `masks` | Fresnel-Zahl und Beugung am Quadrat (mit Fresnel-Integralen), Talbot-Teppich, Grenzen der Fresnel-Näherung, evaneszente Wellen, Wrap-around und Padding, Transferfunktion vs. Impulsantwort, band-limitierte Propagation, Airy-Scheibchen, axiale Intensität (Mahajan), Poisson-Fleck |
| [02_gaussian_beams](02_gaussian_beams.ipynb) | `beams`, `propagators`, `masks.thin_lens` | Strahlradius, Krümmung und Gouy-Phase, Vergleich Analytik/Numerik, Gouy-Phase höherer Moden, HG- und LG-Modengalerie, rotierende Modenüberlagerung, Fokus einer Linse |
| [03_interfaces_thin_films](03_interfaces_thin_films.ipynb) | `interfaces`, `thin_films` | Fresnel-Koeffizienten, Brewster-Winkel, Totalreflexion mit evaneszenter Welle, Metallreflexion, Seifenhautfarben, Antireflexschicht, Bragg-Spiegel, kohärente vs. inkohärente dicke Platten, dünne Metallfilme |
| [04_polarization](04_polarization.ipynb) | `polarization.jones`, `polarization.mueller` | Polarisationsellipsen und Stokes-Parameter, Malus' Gesetz und Polarisatorketten, Wellenplatten auf der Poincaré-Kugel, Depolarisation, Polarisation durch Reflexion, Fresnel-Rhombus, Metallreflexion |
| [05_imaging_zernikes](05_imaging_zernikes.ipynb) | `imaging`, `bases`, `propagators` | Fringe-Zernike-Galerie, PSF mit Aberrationen, Strehl-Verhältnis und Maréchal-Näherung, Zernike-Fit, kohärente vs. inkohärente Abbildung (Siemensstern, CTF/OTF), Rayleigh-Kriterium, 3D-Fokus |
| [06_TIE](06_TIE.ipynb) | `TIE`, `utils.get_z_derivative` | Phasenobjekte durch Defokus sichtbar machen, TIE vorwärts und rückwärts, Rauschen vs. Defokus-Abstand, Regularisierung, Savitzky-Golay-Ableitung, iterative Verbesserung |
| [07_IFTA](07_IFTA.ipynb) | `ifta` | Strahlformung mit Gerchberg-Saxton, Konvergenz, Hybrid-Output-Output, Einfluss der Startphase, Phasenquantisierung und Zwillingsbild |
| [08_WPM](08_WPM.ipynb) | `wpm` | Wave Propagation Method: Test im homogenen Medium, GRIN-Medium, Stufenindex-Wellenleiter, dicke Zylinderlinse mit sphärischer Aberration |

Alle Notebooks verwenden die Konvention von pyoptics, $\exp(i(\mathbf{k}\cdot\mathbf{r} - \omega t))$
(siehe das Haupt-README).

## Ausführen

Voraussetzung ist eine Installation von pyoptics wie im Haupt-README
beschrieben, dazu matplotlib und Jupyter:

```bash
pip install matplotlib jupyterlab
jupyter lab notebooks/
```

Die Rechenzeiten liegen auf einem normalen Laptop zwischen wenigen Sekunden
und etwa einer Minute pro Notebook (am längsten: 01 und 08).

Um alle Notebooks neu auszuführen und die Ausgaben zu speichern:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
```
