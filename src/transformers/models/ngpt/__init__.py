# Copyright 2024 The HuggingFace Team. All rights reserved.
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

from typing import TYPE_CHECKING

from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_keras_nlp_available,
    is_tensorflow_text_available,
    is_tf_available,
    is_torch_available,
)


_import_structure = {
    "configuration_ngpt": ["NgptConfig", "NgptOnnxConfig"],
}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_ngpt"] = [
        "NgptDoubleHeadsModel",
        "NgptForQuestionAnswering",
        "NgptForSequenceClassification",
        "NgptForTokenClassification",
        "NgptLMHeadModel",
        "NgptModel",
        "NgptPreTrainedModel",
        "load_tf_weights_in_ngpt",
    ]

try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_tf_ngpt"] = [
        "TFNgptDoubleHeadsModel",
        "TFNgptForSequenceClassification",
        "TFNgptLMHeadModel",
        "TFNgptMainLayer",
        "TFNgptModel",
        "TFNgptPreTrainedModel",
    ]

try:
    if not is_keras_nlp_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_ngpt_tf"] = ["TFGPT2Tokenizer"]

if TYPE_CHECKING:
    from .configuration_ngpt import NgptConfig, NgptOnnxConfig

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_ngpt import (
            NgptDoubleHeadsModel,
            NgptForQuestionAnswering,
            NgptForSequenceClassification,
            NgptForTokenClassification,
            NgptLMHeadModel,
            NgptModel,
            NgptPreTrainedModel,
            load_tf_weights_in_ngpt,
        )

    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_tf_ngpt import (
            TFNgptDoubleHeadsModel,
            TFNgptForSequenceClassification,
            TFNgptLMHeadModel,
            TFNgptMainLayer,
            TFNgptModel,
            TFNgptPreTrainedModel,
        )

    try:
        if not is_keras_nlp_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        pass
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
