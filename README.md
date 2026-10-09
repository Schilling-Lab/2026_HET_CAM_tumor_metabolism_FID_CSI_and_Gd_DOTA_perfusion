# hypermri 🧲

**Schilling-AG python module to work with Bruker MRI data.**

HyperMRI focuses on repeatable reconstruction, spectroscopy, and visualization workflows built around **xarray** datasets. It provides a robust interface for handling complex multi-dimensional MRI data, bridging the gap between raw vendor formats and high-level analysis.

## Features

*   **Vendor Agnostic Core**: specialized handling for `Bruker`, with architecture supporting `Siemens` and others.
*   **Xarray Integration**: Data is loaded into `xarray.DataArray` and `xarray.Dataset` objects, preserving rich metadata and coordinates (read, phase, slice, repetition, etc.).
*   **Advanced Visualization**: Built-in plotters for complex, multi-dimensional MRI/CSI data.
*   **Spectroscopy & Imaging**: Unified handling of both spectroscopic (CSI, single voxel) and anatomical imaging data.
*   **Reproducibility**: Tools for repeatable reconstruction and fitting workflows.



## Documentation

Detailed documentation is available in the `docs/` folder. Since it is not yet hosted online, you can build and view it locally.


### Key Guides

*   **[User Guide](https://gitlab.lrz.de/mri_nuk/python/python-code-bruker-7t/-/blob/main/docs/USER_GUIDE.md)**: Installation, usage examples, and core concepts.
*   **[Developer Guide](https://gitlab.lrz.de/mri_nuk/python/python-code-bruker-7t/-/blob/main/docs/DEVELOPER_GUIDE.md)**: Setup, testing, and release workflows.

## Installation

We strictly recommend using **[Anaconda](https://anaconda.org/)** (or Miniconda) to manage dependencies.

**Prerequisites:**
* **Git**: Required to download specific dependencies. [Install Git](https://git-scm.com/downloads).
* **Git LFS**: Required to download tutorial media files. [Install Git LFS](https://git-lfs.github.com/).

1. **Create (anaconda) environment**
```bash
conda create -n hypermri_dev python=3.10
conda activate hypermri_dev
conda install git git-lfs
git lfs install
```

2. **clone git-repository**
```bash
git clone https://gitlab.lrz.de/mri_nuk/python/python-code-bruker-7t.git
cd python-code-bruker-7t
```

3. **Install package** including documentation
```bash
pip install -e .[docs]
```
in case you get a "no matches found: .[docs]" try:

```bash
pip install -e ".[docs]"
```

4. **Pull documentation media and build HTML**:
```bash
# Download large media files (images/videos)
git lfs pull --include "docs/source/_static/media/**"

cd docs
make html
```

5.  **View documentation**:
Open `docs/build/html/index.html` in your web browser.


## Usage

```python
from hypermri import Study

# Load a study folder
study = Study("path/to/study/folder")

# Access and plot an experiment
exp = study[1]
exp.plot()
```

See the [User Guide](https://gitlab.lrz.de/mri_nuk/python/python-code-bruker-7t/-/blob/main/docs/USER_GUIDE.md) for more examples.

## Contributing

We welcome contributions! Please see the [Developer Guide](https://gitlab.lrz.de/mri_nuk/python/python-code-bruker-7t/-/blob/main/docs/DEVELOPER_GUIDE.md) for instructions on setting up your environment and running tests.

## License

MIT License. See [LICENSE](https://gitlab.lrz.de/mri_nuk/python/python-code-bruker-7t/-/blob/main/LICENSE) for details.
