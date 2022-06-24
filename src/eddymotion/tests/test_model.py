# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2021 The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
"""Unit tests exercising models."""
import os.path as op
import numpy as np
import pytest

from eddymotion import model
from eddymotion.dmri import DWI


def test_trivial_model():
    """Check the implementation of the trivial B0 model."""

    # Should not allow initialization without a B0
    with pytest.raises(ValueError):
        model.TrivialB0Model(gtab=np.eye(4))

    _S0 = np.random.normal(size=(10, 10, 10))

    tmodel = model.TrivialB0Model(gtab=np.eye(4), S0=_S0)

    assert tmodel.fit() is None

    assert np.all(_S0 == tmodel.predict((1, 0, 0)))


def test_average_model():
    """Check the implementation of the average DW model."""

    gtab = np.array(
        [
            [-0.31, 0.933, 0.785, 25],
            [0.25, 0.565, 0.21, 500],
            [-0.861, -0.464, 0.564, 1000],
            [0.307, -0.766, 0.677, 1000],
            [0.736, 0.013, 0.774, 2000],
        ]
    ).T
    data = np.random.randn(10, 10, 10, gtab.shape[1])
    data_w25 = data[..., 1:]
    data_1000 = data[..., 2:4]
    data_2000 = data[..., 2:]

    tmodel_mean = model.AverageDWModel(gtab=gtab, bias=False, stat="mean")
    tmodel_median = model.AverageDWModel(gtab=gtab, bias=False, stat="median")
    tmodel_1000 = model.AverageDWModel(
        gtab=gtab, bias=False, th_high=1000, th_low=1000
    )
    tmodel_2000 = model.AverageDWModel(
        gtab=gtab, bias=False, th_high=2000, th_low=1000
    )

    # Verify that fit function returns nothing
    assert tmodel_mean.fit(data) is None

    tmodel_median.fit(data)
    tmodel_1000.fit(data)
    tmodel_2000.fit(data)

    # Verify that the right statistics is applied and that the model discard b-values < 50
    assert np.all(tmodel_mean.predict() == np.mean(data_w25, axis=-1))
    assert np.all(tmodel_median.predict() == np.median(data_w25, axis=-1))

    # Verify that the threshold for b-value selection works as expected
    assert np.all(tmodel_1000.predict() == np.median(data_1000, axis=-1))
    assert np.all(tmodel_2000.predict() == np.median(data_2000, axis=-1))


def test_two_initialisations(pkg_datadir):
    """Check that the two different initialisations result in the same models"""

    # Load test data
    dmri_dataset = DWI.from_filename(op.join(pkg_datadir, "dwi.h5"))

    # Split data into test and train set
    data_train, data_test = dmri_dataset.logo_split(10)

    # Direct initialisation
    model1 = model.AverageDWModel(
        dmri_dataset.gradients,
        S0=dmri_dataset.bzero,
        th_low=100,
        th_high=1000,
        bias=True,
        stat="mean",
    )
    model1.fit(data_train[0])
    predicted1 = model1.predict(data_test[1])

    # Initialisation via ModelFactory
    model2 = model.ModelFactory.init(
        gtab=data_train[1],
        model="avg",
        S0=dmri_dataset.bzero,
        th_low=100,
        th_high=1000,
        bias=True,
        stat="mean",
    )
    model2.fit(data_train[0])
    predicted2 = model2.predict(data_test[1])

    assert predicted1 == predicted2
