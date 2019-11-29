/*
variables
*/
var model;
var canvas = null;
var classNames = [];
var lastCoord = null;
var userStroke = []; // The stroke drawn by user.
var predictStroke = []; // The stroke predicted by model.
var mousePressed = false;
var mode;
var currentModel = "flower"; // The current model, flower or school_bus model.

const CANVAS_SIZE = [500, 500];
const DRAW_SIZE = [255, 255];
const DRAW_PREC = 0.01;
const MAX_LEN = 352;
const RAINBOW_COLORS = ["#FF0000", "#FF7F00", "#FFFF00", "#00FF00", "#00FFFF", "#0000FF", "#8B00FF"];

/*
load the model
*/
async function start(cur_mode, model) {
    // arabic or english
    mode = cur_mode;
    currentModel = model;

    // load drawing canvas
    if (canvas === null) {
        loadCanvas();
        // Allow drawing.
        await allowDrawing(currentModel);
    } else {
        await erase();
    }
}

/*
prepare the drawing canvas 
*/
function loadCanvas() {
    console.log("prepare the drawing canvas.");
    canvas = window._canvas = new fabric.Canvas('canvas');
    canvas.backgroundColor = '#ffffff';
    canvas.isDrawingMode = 0;
    canvas.freeDrawingBrush.color = "black";
    canvas.freeDrawingBrush.width = 3;
    canvas.renderAll();
    // setup listeners
    canvas.on('mouse:up', function (e) {
        console.log("mouse:up");
        if (canvas.isDrawingMode === 1) {
            mousePressed = false;
            canvas.isDrawingMode = 0;
            // Auto draw with starting inks.
            autodraw();
        }
    });
    canvas.on('mouse:down', function (e) {
        console.log("mouse:down.");
        mousePressed = true;
    });
    canvas.on('mouse:move', function (e) {
        if (canvas.isDrawingMode === 1) {
            recordCoor(e);
        }
    });
}

/*
record the current drawing coordinates
*/
function recordCoor(event) {
    if (mousePressed) {
        var pointer = canvas.getPointer(event.e);
        var posX = pointer.x;
        var posY = pointer.y;
        userStroke.push([posX, posY]);

        if (lastCoord !== null) {
            // Calc the delta.
            let xDelta = posX - lastCoord[0];
            let yDelta = lastCoord[1] - posY; // Reverse the y coordinate.

            // Normalization.
            // xDelta = Number((xDelta / DRAW_SIZE[0]).toFixed(2));
            // yDelta = Number((yDelta / DRAW_SIZE[1]).toFixed(2));
            xDelta = xDelta / DRAW_SIZE[0];
            yDelta = yDelta / DRAW_SIZE[1];

            if (predictStroke.length > 0) {
                if (xDelta === 0.0 && predictStroke[predictStroke.length - 1][0] === 0.0) {
                    // Merge if only move in y axis.
                    predictStroke[predictStroke.length - 1][1] += yDelta;
                    lastCoord = [posX, posY];
                    return;
                }

                if (yDelta === 0.0 && predictStroke[predictStroke.length - 1][1] === 0.0) {
                    // Merge if only move in x axis.
                    predictStroke[predictStroke.length - 1][0] += xDelta;
                    lastCoord = [posX, posY];
                    return;
                }
            }

            // Ignore < DRAW_PREC.
            if (Math.abs(xDelta) >= DRAW_PREC || Math.abs(yDelta) >= DRAW_PREC) {
                predictStroke.push([xDelta, yDelta, 0.0, 0.0]);
                lastCoord = [posX, posY];
            }

            // console.log(predictStroke)
        } else {
            lastCoord = [posX, posY];
        }
    }
}

/*
allow drawing on canvas
*/
async function allowDrawing(currentModel) {
    // load the model
    model = await tf.loadLayersModel('model/' + currentModel + '/model.json');

    canvas.isDrawingMode = 1;
    if (mode === 'en')
        document.getElementById('status').innerHTML = 'Model Loaded';
    $('button').prop('disabled', false);
    var slider = document.getElementById('myRange');

    // Set the canvas drawing brush width with the input of slider.
    slider.oninput = function () {
        // console.log("Current drawing brush width: " + this.value);
        canvas.freeDrawingBrush.width = this.value;
    };
}

/*
clear the canvs 
*/
async function erase() {
    canvas.clear();
    canvas.backgroundColor = '#ffffff';
    userStroke = [];
    predictStroke = [];
    lastCoord = null;

    if (mode === 'en')
        document.getElementById('status').innerHTML = 'Loading Model... ';

    // allow drawing on the canvas
    await allowDrawing(currentModel);
}

/*
preprocess the data
*/
function preprocess(inks) {
    return tf.tidy(() => {
        //convert to a tensor
        let tensor = tf.tensor(inks);

        return tensor.expandDims(0);
    })
}

/*
Auto draw.
*/
function autodraw() {
    if (model === undefined) {
        console.log("Model unloaded.!");
        return;
    }

    // predictStroke = [[0.1, 0.04, 0., 0.], [0.12, -0., 0., 0.], [0.04, -0.02, 0., 0.], [0.03, -0.04, 0., 0.],
    //     [-0., -0.07, 0., 0.], [-0.02, -0.02, 0., 0.], [-0.11, -0.07, 0., 0.], [-0.14, -0.01, 0., 0.],
    //     [-0.04, 0.02, 0., 0.], [-0.04, 0.04, 0., 0.]];

    // predictStroke = [[0.07, 0.01, 0., 0.], [0.04, -0.03, 0., 0.], [0.04, -0.06, 0., 0.], [0.01, -0.08, 0., 0.],
    //     [-0.01, -0.03, 0., 0.], [-0.06, -0.02, 0., 0.], [-0.11, -0.01, 0., 0.], [-0.06, 0.07, 0., 0.],
    //     [-0.02, 0.11, 0., 0.], [0.09, 0.03, 0., 0.]];
    console.log("The user stroke inks:");
    console.log(userStroke);

    console.log("The input inks:");
    console.log(predictStroke);
    // The initial inks len.
    const initialLen = predictStroke.length;
    console.log("The initial inks len: " + initialLen);
    // Enter the initial inks.
    let pred = model.predict(preprocess(predictStroke)).dataSync();
    // Find he last ink.
    const index = (initialLen - 1) * 4;
    let pred_ = [pred[index], pred[index + 1], pred[index + 2], pred[index + 3]];
    // Save the new ink.
    predictStroke.push(pred_);
    // Pred the left inks.
    let inp = null;
    do {
        // Use the last ink as input.
        inp = [predictStroke[predictStroke.length - 1]];
        // Enter the initial inks.
        pred = model.predict(preprocess(inp)).dataSync();
        // Find he last ink.
        pred_ = [pred[0], pred[1], pred[2], pred[3]];
        // Save the new ink.
        predictStroke.push(pred_);
    } while (pred[3] < 0.5 && predictStroke.length <= MAX_LEN - initialLen);
    console.log(predictStroke);
    // Pop the initial inks.
    predictStroke.splice(0, initialLen);
    // Draw predict inks with the begin position of user stroke end.
    drawInks(predictStroke, "green", userStroke[userStroke.length - 1]);
    // drawInks(predictStroke, "green", [150, 150]);
}

/*
Drawing with inks.
*/
function drawInks(inks, color, beginPos) {
    // 1, generate coords by deltas.
    let inkCoords = [[beginPos[0] / DRAW_SIZE[0], beginPos[1] / DRAW_SIZE[1], 0, 0]];
    for (let ink in inks) {
        // console.log(predictStroke[ink]);
        let posX = inkCoords[ink][0] + inks[ink][0];
        let posY = inkCoords[ink][1] - inks[ink][1];
        let endFlag = inks[ink][2];
        let completeFlag = inks[ink][3];
        inkCoords.push([posX, posY, endFlag, completeFlag]);
    }
    // 2, zoom in to DRAW_SIZE scale.
    for (let ink in inkCoords) {
        inkCoords[ink][0] *= DRAW_SIZE[0];
        inkCoords[ink][1] *= DRAW_SIZE[1];
    }
    // console.log(inkCoords);
    // 3. Draw every stroke.
    let stroke = [];
    for (let ink in inkCoords) {
        // Check if is complete ink.
        if (inkCoords[ink][3] > 0.5) {
            drawStroke(stroke);
            stroke = [];
            return;
        }
        // Check if is stroke end ink.
        if (inkCoords[ink][2] > 0.5) {
            // It's the stroke end ink, draw current stroke.
            stroke.push([inkCoords[ink][0], inkCoords[ink][1]]);
            drawStroke(stroke);
            stroke = [];
        } else {
            // It's one point in stroke, add into current stroke.
            stroke.push([inkCoords[ink][0], inkCoords[ink][1]]);
        }
    }
    if (stroke.length !== 0) {
        // There has been left inks.
        drawStroke(stroke);
    }
}

/*
Draw stroke.
*/
function drawStroke(stroke) {
    let pathStr = "M " + stroke[0][0] + " " + stroke[0][1]; // The start ink.
    for (let ink in stroke) {
        if (ink == 0) {
            continue;
        }
        pathStr = pathStr + " L " + stroke[ink][0] + " " + stroke[ink][1];
    }
    console.log("pathStr: " + pathStr);
    const path = new fabric.Path(pathStr);
    const randColorInd = Math.round(Math.random() * 6);
    path.set({fill: "transparent", stroke: RAINBOW_COLORS[randColorInd]});
    canvas.add(path);
}