"""Object catalogue and the mapping between UI names and model labels.

The embedded YOLOX model was trained on the 80 COCO classes.  The UI presents
friendlier names ("Smartphone") and this module translates them to the label
the model actually knows ("cell phone").  Remote backends can work with any
free-text label, so the catalogue also exposes a custom target.
"""

from __future__ import annotations

from dataclasses import dataclass

CUSTOM_OBJECT_KEY = "custom"
DEFAULT_OBJECT_KEY = "smartphone"


@dataclass(frozen=True)
class ObjectSpec:
    key: str
    label: str
    model_label: str
    emoji: str = ""

    @property
    def display(self) -> str:
        return f"{self.emoji}  {self.label}".strip()

    @property
    def supported_locally(self) -> bool:
        return self.model_label.strip().lower() in COCO_LABEL_SET


COCO_LABELS: tuple[str, ...] = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)

COCO_LABEL_SET = frozenset(COCO_LABELS)


COCO_OBJECTS: tuple[ObjectSpec, ...] = (
    ObjectSpec("smartphone", "Smartphone", "cell phone", "📱"),
    ObjectSpec("laptop", "Laptop", "laptop", "💻"),
    ObjectSpec("book", "Book", "book", "📚"),
    ObjectSpec("person", "Person", "person", "🧍"),
    ObjectSpec("cat", "Cat", "cat", "🐈"),
    ObjectSpec("dog", "Dog", "dog", "🐕"),
    ObjectSpec("car", "Car", "car", "🚗"),
    ObjectSpec("bicycle", "Bicycle", "bicycle", "🚲"),
    ObjectSpec("bottle", "Bottle", "bottle", "🍶"),
    ObjectSpec("cup", "Cup", "cup", "☕"),
    ObjectSpec("chair", "Chair", "chair", "🪑"),
    ObjectSpec("keyboard", "Keyboard", "keyboard", "⌨️"),
    ObjectSpec("mouse", "Mouse", "mouse", "🖱️"),
    ObjectSpec("tv", "TV", "tv", "📺"),
    ObjectSpec("backpack", "Backpack", "backpack", "🎒"),
    ObjectSpec("handbag", "Handbag", "handbag", "👜"),
    ObjectSpec("clock", "Clock", "clock", "🕐"),
    ObjectSpec("vase", "Vase", "vase", "🏺"),
    ObjectSpec("scissors", "Scissors", "scissors", "✂️"),
    ObjectSpec("toothbrush", "Toothbrush", "toothbrush", "🪥"),
)

OBJECT_BY_KEY: dict[str, ObjectSpec] = {obj.key: obj for obj in COCO_OBJECTS}
OBJECT_BY_MODEL_LABEL: dict[str, ObjectSpec] = {
    obj.model_label: obj for obj in COCO_OBJECTS
}

CUSTOM_OBJECT = ObjectSpec(
    CUSTOM_OBJECT_KEY,
    "Custom object",
    "",
    "🎯",
)


def find_object(key: str) -> ObjectSpec | None:
    return OBJECT_BY_KEY.get(key)


def find_by_model_label(label: str) -> ObjectSpec | None:
    return OBJECT_BY_MODEL_LABEL.get(label.strip().lower())


def make_custom_target(text: str) -> ObjectSpec:
    text = text.strip()
    return ObjectSpec(CUSTOM_OBJECT_KEY, text, text, "🎯")


def resolve_target(key: str, custom_text: str = "") -> ObjectSpec:
    """Resolve a dropdown value into a target, falling back to custom text."""

    if key == CUSTOM_OBJECT_KEY:
        return make_custom_target(custom_text)
    return find_object(key) or find_object(DEFAULT_OBJECT_KEY) or COCO_OBJECTS[0]