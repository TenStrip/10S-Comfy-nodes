import { app } from "../../scripts/app.js";

const MAX_SLOTS = 12;
const SLOT_FIELDS = ["lora", "strength", "audio_weight", "video_weight"];

function setWidgetVisible(widget, visible) {
    if (!widget) return;

    if (!Object.hasOwn(widget, "_loraStackOriginalComputeSize")) {
        widget._loraStackOriginalComputeSize = widget.computeSize;
        widget._loraStackOriginalDraw = widget.draw;
    }

    widget.hidden = !visible;
    widget.options = widget.options || {};
    widget.options.hidden = !visible;

    if (visible) {
        widget.computeSize = widget._loraStackOriginalComputeSize;
        widget.draw = widget._loraStackOriginalDraw;
        if (widget.element) widget.element.style.display = "";
    } else {
        widget.computeSize = () => [0, -4];
        widget.draw = () => {};
        if (widget.element) widget.element.style.display = "none";
    }
}

function setupNode(node) {
    if (node._loraStackVisibilitySetup) return;

    const countWidget = node.widgets?.find(widget => widget.name === "num_loras");
    if (!countWidget) return;

    node._loraStackVisibilitySetup = true;

    const updateVisibility = () => {
        const count = Math.max(1, Math.min(MAX_SLOTS, Number(countWidget.value) || 1));
        for (let i = 1; i <= MAX_SLOTS; i++) {
            const visible = i <= count;
            for (const field of SLOT_FIELDS) {
                setWidgetVisible(
                    node.widgets.find(widget => widget.name === `${field}_${i}`),
                    visible,
                );
            }
        }

        node.setSize([node.size[0], node.computeSize()[1]]);
        node.setDirtyCanvas(true, true);
    };

    const originalCountCallback = countWidget.callback;
    countWidget.callback = function () {
        const result = originalCountCallback?.apply(this, arguments);
        updateVisibility();
        return result;
    };

    const originalOnConfigure = node.onConfigure;
    node.onConfigure = function () {
        const result = originalOnConfigure?.apply(this, arguments);
        setTimeout(updateVisibility, 0);
        return result;
    };

    node._loraStackUpdateVisibility = updateVisibility;
    setTimeout(updateVisibility, 0);
}

app.registerExtension({
    name: "10SNodes.LTXLoraStackAV",

    nodeCreated(node) {
        if (node.comfyClass === "LTXLoraStackAV" || node.type === "LTXLoraStackAV") {
            setupNode(node);
        }
    },
});
