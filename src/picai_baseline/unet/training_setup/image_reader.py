# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Callable, Optional, Sequence
import numpy as np
from torch.utils.data import Dataset
import SimpleITK as sitk
from monai.config import DtypeLike
from monai.transforms import Randomizable, apply_transform
from monai.utils import MAX_SEED, get_seed
from picai_baseline.unet.training_setup.preprocess_utils import z_score_norm
try:
    import numpy.typing as npt
except ImportError:  # pragma: no cover
    pass


class SimpleITKDataset(Dataset, Randomizable):
    """
    Loads image/segmentation pairs of files from the given filename lists. Transformations can be specified
    for the image and segmentation arrays separately.
    The difference between this dataset and `ArrayDataset` is that this dataset can apply transform chain to images
    and segs and return both the images and metadata, and no need to specify transform to load images from files.
    For more information, please see the image_dataset demo in the MONAI tutorial repo,
    https://github.com/Project-MONAI/tutorials/blob/master/modules/image_dataset.ipynb

    Also performs instance-wise z-score normalization of all MRI sequences before concatenation.
    """

    def __init__(
        self,
        image_files: Sequence[str],
        seg_files: Optional[Sequence[str]] = None,
        labels: Optional[Sequence[float]] = None,
        transform: Optional[Callable] = None,
        seg_transform: Optional[Callable] = None,
        dtype: DtypeLike = np.float32,

    ) -> None:
        """
        Initializes the dataset with the image and segmentation filename lists. The transform `transform` is applied
        to the images and `seg_transform` to the segmentations.

        Args:
            image_files: list of image filenames
            seg_files: if in segmentation task, list of segmentation filenames
            labels: if in classification task, list of classification labels
            transform: transform to apply to image arrays
            seg_transform: transform to apply to segmentation arrays
            dtype: if not None convert the loaded image to this data type

        Raises:
            ValueError: When ``seg_files`` length differs from ``image_files``

        """

        if seg_files is not None and len(image_files) != len(seg_files):
            raise ValueError(
                "Must have same the number of segmentation as image files: "
                f"images={len(image_files)}, segmentations={len(seg_files)}."
            )

        self.image_files = image_files
        self.seg_files = seg_files
        self.labels = labels
        self.transform = transform
        self.seg_transform = seg_transform
        self.set_random_state(seed=get_seed())
        self._seed = 0  # transform synchronization seed

    def __len__(self) -> int:
        return len(self.image_files)

    def randomize(self, data: Optional[Any] = None) -> None:
        self._seed = self.R.randint(MAX_SEED, dtype="uint32")

    def prepare_scan(self, path: str) -> "npt.NDArray[Any]":
        img = sitk.ReadImage(path)
        return np.expand_dims(
            sitk.GetArrayFromImage(
                img
            ).astype(np.float32), axis=(0, 1)
        )

    def read_meta_data(self, path: str, metas, medians, means, stds):
        """
        Reads the values of the requested meta data from the image file.

        Args:
        path: string of path to the image
        metas: list of names of meta data values
        medians: list of default values for the meta data
        means: list of mean values for the meta data after filling in missing values and applying the log1p transformation
        stds: list of standard deviation values for the meta data after filling in missing values and applying the log1p transformation

        Returns:
        meta_data: list of meta data values, where missing values are computed
        when possible and otherwise set to the population median.
        The log is taken of PSAD, PSA, and prostate volume.
        """
        img = sitk.ReadImage(path)
        meta_data = []
        for i in range(len(metas)):
            try:
                md = img.GetMetaData(metas[i])
                if metas[i] == "0010|1010":
                    meta_data.append(int(md[:3]))
                else:
                    meta_data.append(float(md))
            except:  # if the value is not in the meta data or it is not a number
                meta_data.append(None)
        meta_data = self.fill_in_missing(meta_data, metas, medians)
        meta_data = self.log_values(meta_data, metas)
        meta_data = self.normalize_values(meta_data, metas, means, stds)
        return np.numpy(meta_data)

    def normalize_values(self, meta_data, metas, means, stds):
        """
        Normalizes all variables.

        Args:
        meta_data: list of the raw values of the meta data, with None for missing data
        metas: list of names of meta data values
        means: list of means for the data
        stds: list of standard deviations for the data

        Returns:
        meta_data: the normalized input meta_data 
        """

        
        for i in range(len(meta_data)):
            meta_data[i] = (meta_data[i] - means[i]) / stds[i]
        return meta_data
    
    def log_values(self, meta_data, metas):
        """
        Takes the log of (x + 1) for values of PSAD, PSA, and prostate volume.
        It assumes the values for each of those are given.

        Args:
        meta_data: list of the raw values of the meta data
        metas: list of the names of the meta data

        Returns:
        meta_data: the input meta_data with the log-transformed PSAD, PSA, and prostate volume
        """
        index_psad = metas.index("PSAD_REPORT")
        index_psa = metas.index("PSA_REPORT")
        index_pro_vol = metas.index("PROSTATE_VOLUME_REPORT")
        for i in [index_psad, index_psa, index_pro_vol]:
            meta_data[i] = np.log1p(meta_data[i])
        return meta_data
    
    def fill_in_missing(self, meta_data, metas, medians):
        """
        Fills in missing meta data by computing the value if possible, and otherwise filling in the median.

        Args:
        meta_data: list of the raw values of the meta data, with None for missing data
        metas: list of names of meta data values
        medians: list of default values for the meta data

        Returns:
        meta_data: the input meta_data with all None-values filled in by computation or median
        """
        index_psad = metas.index("PSAD_REPORT")
        index_psa = metas.index("PSA_REPORT")
        index_pro_vol = metas.index("PROSTATE_VOLUME_REPORT")
        if meta_data[index_psad] is None:
            if meta_data[index_psa] is not None and meta_data[index_pro_vol] is not None and meta_data[index_pro_vol] != 0:
                meta_data[index_psad] = meta_data[index_psa] / meta_data[index_pro_vol]
        if meta_data[index_psa] is None:
            if meta_data[index_psad] is not None and meta_data[index_pro_vol] is not None:
                meta_data[index_psa] = meta_data[index_psad] * meta_data[index_pro_vol]
        if meta_data[index_pro_vol] is None:
            if meta_data[index_psa] is not None and meta_data[index_psad] is not None and meta_data[index_psad] != 0:
                meta_data[index_pro_vol] = meta_data[index_psa] / meta_data[index_psad]
        for i in range(len(meta_data)):
            if meta_data[i] is None:
                meta_data[i] = medians[i]
        return meta_data

    def __getitem__(self, index: int):
        self.randomize()
        seg, label = None, None

        # load all sequences (work-around) and optionally meta
        img_t2w = z_score_norm(self.prepare_scan(str(self.image_files[index][0])), 99.5)
        img_adc = z_score_norm(self.prepare_scan(str(self.image_files[index][1])), 99.5)
        img_hbv = z_score_norm(self.prepare_scan(str(self.image_files[index][2])), 99.5)

        img = np.concatenate([img_t2w, img_adc, img_hbv], axis=1)

        meta_data = self.read_meta_data(str(self.image_files[index][0]),
                                        ["0010|1010", "PSAD_REPORT", "PSA_REPORT", "PROSTATE_VOLUME_REPORT"],
                                        [66, 8.5, 0.15, 57.], # medians computed from training population after filling in computable values
                                        [65.595333, 2.300051, 0.184540, 4.059110], # means computed from training population after filling in computable values and applying log1p transformation
                                        [7.191527, 0.639685, 0.187901, 0.512711])  # standard deviations computed from training population after filling in computable values and applying log1p transformation

        if self.seg_files is not None:
            seg = sitk.GetArrayFromImage(sitk.ReadImage(self.seg_files[index])).astype(np.int8)
            seg = np.expand_dims(seg, axis=(0, 1))

        # apply the transforms
        if self.transform is not None:
            img = apply_transform(self.transform, img, map_items=False)

        if self.seg_transform is not None:
            seg = apply_transform(self.seg_transform, seg, map_items=False)

        if self.labels is not None:
            label = self.labels[index]

        # construct outputs
        data = [img, meta_data]
        if seg is not None:
            data.append(seg)
        if label is not None:
            data.append(label)
        if len(data) == 1:
            return data[0]

        # use tuple instead of list as the default collate_fn callback of MONAI DataLoader flattens nested lists
        return tuple(data)
