# Dataset

## User-defined dataset (main workflow)

You choose the species that the model recognizes. Create one folder per species
and put labelled bird images inside it:

```text
data/user_defined/
├── Indian_Roller/
│   ├── roller_01.jpg
│   └── roller_02.jpg
├── House_Sparrow/
│   ├── sparrow_01.jpg
│   └── sparrow_02.jpg
└── Rose_Ringed_Parakeet/
    └── parakeet_01.jpg
```

Use at least 20 images per species (50+ is better) and at least two species.
Keep each image only in its correct species folder. The folder names become the
labels shown in predictions. Then run:

```bash
python scripts/prepare_custom_dataset.py
python -m src.train
```

## Optional public benchmark

`download_cub.py` and `prepare_dataset.py` are retained if you want to compare
your custom dataset against CUB-200-2011. The archive is not committed because
it is large and has its own usage terms.

## Supplied `archive` dataset

The project also supports `data/archive/Birds_25`, with its existing `train`
and `valid` folders for the specified 25 Indian bird species. Do **not** mix
the validation images into training. Prepare this dataset with:

```bash
python scripts/prepare_birds25_dataset.py --train-per-class 100 --valid-per-class 30
```

Use every readable image in the supplied `train` and `valid` directories with
`python scripts/prepare_birds25_dataset.py --all`.
