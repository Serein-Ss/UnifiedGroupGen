# C2DB-51

C2DB-24 contains 16k 2D materials

Carbon-24 contains 10k carbon materials, which share the same composition, but have different structures. There is 1 element and the materials have 6 - 51 atoms in the unit cells.

## What is in the dataset?

Carbon-24 includes various carbon structures obtained via *ab initio* random structure searching (AIRSS) (Pickard & Needs, 2006; 2011) performed at 10 GPa.

## Stabiity of curated materials

The original dataset includes 101529 carbon structures, and we selected the 10% of the carbon structure with the lowest energy per atom to create Carbon-24. All 10153 structures are at local energy minimum after DFT relaxation. The most stable structure is diamond at 10 GPa. All remaining structures are thermodynamically unstable but may be kinetically stable.

## Visualization of structures

<p align="center">
  <img src="../../assets/carbon_24.png" />
</p>
## Citation

Please consider citing the following paper:

```
@article{haastrup2018computational,
  title={The Computational 2D Materials Database: high-throughput modeling and discovery of atomically thin crystals},
  author={Haastrup, Sten and Strange, Mikkel and Pandey, Mohnish and Deilmann, Thorsten and Schmidt, Per S and Hinsche, Nicki F and Gjerding, Morten N and Torelli, Daniele and Larsen, Peter M and Riis-Jensen, Anders C and others},
  journal={2D Materials},
  volume={5},
  number={4},
  pages={042002},
  year={2018},
  publisher={IOP Publishing}
}
```

```
@article{gjerding2021recent,
  title={Recent progress of the computational 2D materials database (C2DB)},
  author={Gjerding, Morten Niklas and Taghizadeh, Alireza and Rasmussen, Asbj{\o}rn and Ali, Sajid and Bertoldo, Fabian and Deilmann, Thorsten and Kn{\o}sgaard, Nikolaj R{\o}rb{\ae}k and Kruse, Mads and Larsen, Ask Hjorth and Manti, Simone and others},
  journal={2D Materials},
  volume={8},
  number={4},
  pages={044002},
  year={2021},
  publisher={IOP Publishing}
}
```







