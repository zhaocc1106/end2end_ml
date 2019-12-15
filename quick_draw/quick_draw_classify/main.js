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

const DRAW_SIZE = [255, 255];
const DRAW_PREC = 0.03;

/*
load the model
*/
async function start(cur_mode) {
    // arabic or english
    mode = cur_mode;

    // load drawing canvas
    if (canvas === null) {
        loadCanvas();
        // Allow drawing.
        allowDrawing();
    } else {
        await erase();
    }

    // Load class names text.
    await loadDict();

    // load the model
    model = await tf.loadLayersModel('model/model.json');

    // warm up
    let prediction = model.predict(tf.zeros([1, 30, 3]));
    prediction.print();
    prediction.dispose();
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
    canvas.freeDrawingBrush.width = 5;
    canvas.renderAll();
    // setup listeners
    canvas.on('mouse:up', function (e) {
        console.log("mouse:up");
        if (canvas.isDrawingMode === 1) {
            mousePressed = false;

            // Classify current quick draw.
            // classify();
            // The last ink is end of stroke.
            if (predictStroke.length > 0) {
                predictStroke[predictStroke.length - 1][2] = 1.0;
            }
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
load the class names
*/
async function loadDict() {
    let loc = "model/classes_names";
    if (mode === "zh") {
        loc = 'model/classes_names_zh';
    }

    await $.ajax({
        url: loc,
        dataType: 'text',
    }).done(success);
}

/*
load the class names
*/
function success(data) {
    const lst = data.split(/\n/);
    for (let i = 0; i < lst.length - 1; i++) {
        classNames[i] = lst[i];
    }
    console.log(classNames);
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
            xDelta = Number((xDelta / DRAW_SIZE[0]).toFixed(8));
            yDelta = Number((yDelta / DRAW_SIZE[1]).toFixed(8));
            // xDelta = xDelta / DRAW_SIZE[0];
            // yDelta = yDelta / DRAW_SIZE[1];

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
                predictStroke.push([xDelta, yDelta, 0.0]);
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
function allowDrawing() {
    canvas.isDrawingMode = 1;
    document.getElementById('status').innerHTML = 'Model Loaded';
    $('button').prop('disabled', false);
    let slider = document.getElementById('myRange');

    // Set the canvas drawing brush width with the input of slider.
    slider.oninput = function () {
        console.log("Current drawing brush width: " + this.value);
        canvas.freeDrawingBrush.width = this.value;
    };
}

/*
clear the canvs 
*/
function erase() {
    canvas.clear();
    canvas.backgroundColor = '#ffffff';
    userStroke = [];
    predictStroke = [];
    lastCoord = null;
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
function classify() {
    if (model === undefined) {
        console.log("Model unloaded.!");
        return;
    }

    console.log("The user stroke inks:");
    console.log(userStroke);

    console.log("The input inks:");
    // The last ink is end of stroke.
    predictStroke[predictStroke.length - 1][2] = 1.0;

    /* A test quick draw. */
    // predictStroke = [[0.1490196, 0.01568627, 0.],
    //     [0.10196081, -0.02745098, 0.],
    //     [0.05098039, -0.03921568, 0.],
    //     [0.03137255, -0.07450981, 0.],
    //     [0.01176471, -0.09411764, 0.],
    //     [-0.04705882, -0.04705882, 0.],
    //     [-0.06274509, -0.03529412, 0.],
    //     [-0.08627453, -0.03137255, 0.],
    //     [-0.14509803, -0.00784314, 0.],
    //     [-0.06666666, 0.03529412, 0.],
    //     [-0.04313725, 0.04705882, 0.],
    //     [-0.01568627, 0.03529412, 0.],
    //     [-0.00392157, 0.09019607, 0.],
    //     [0.03137255, 0.07450981, 0.],
    //     [0.03921568, 0.02352941, 0.],
    //     [0.06274509, 0.00784314, 1.],
    //     [-0.04705882, 0.01960784, 0.],
    //     [-0.09803921, 0.03921569, 0.],
    //     [-0.15686274, 0.09411765, 1.],
    //     [0.42745095, -0.10196079, 0.],
    //     [0.00392157, 0.1137255, 1.],
    //     [0.12941179, -0.12941177, 0.],
    //     [0.03921568, 0.04313726, 0.],
    //     [0.18431373, 0.09803922, 1.],
    //     [-0.14117648, -0.28235295, 0.],
    //     [0.10196078, 0.02352941, 0.],
    //     [0.25490198, 0.00392157, 1.],
    //     [-0.42352942, -0.17647058, 0.],
    //     [0.07843137, -0.07058826, 0.],
    //     [0.16470589, -0.10196078, 1.],
    //     [-0.40000004, 0.12549022, 0.],
    //     [0.00784314, -0.17647061, 1.],
    //     [-0.24705881, 0.26666668, 0.],
    //     [-0., -0.02745098, 0.],
    //     [-0.02352941, -0.03529412, 0.],
    //     [-0.05490196, -0.04313725, 1.]];

    console.log(predictStroke);
    // Predict the class name index.
    let pred = model.predict(preprocess(predictStroke));
    let predArr = pred.dataSync();
    // console.log(predArr.sort().reverse());
    pred.dispose();

    //find the top 5 predictions
    const indices = findIndicesOfMax(predArr, 5);
    console.log("The predict indices:");
    console.log(indices);
    const probs = findTopValues(predArr, 5);
    console.log("The predict probabilities:");
    console.log(probs);
    const names = getClassNames(indices);
    console.log("The predict names:");
    console.log(names);

    //set the table
    setTable(names, probs);
}

/*
get indices of the top probs
*/
function findIndicesOfMax(inp, count) {
    let outp = [];
    for (let i = 0; i < inp.length; i++) {
        outp.push(i); // add index to output array
        if (outp.length > count) {
            outp.sort(function (a, b) {
                return inp[b] - inp[a];
            }); // descending sort the output array
            outp.pop(); // remove the last index (index of smallest element in output array)
        }
    }
    return outp;
}

/*
find the top 5 predictions
*/
function findTopValues(inp, count) {
    let outp = [];
    let indices = findIndicesOfMax(inp, count);
    // show 5 greatest scores
    for (let i = 0; i < indices.length; i++)
        outp[i] = inp[indices[i]];
    return outp
}

/*
get the the class names
*/
function getClassNames(indices) {
    let outp = [];
    for (let i = 0; i < indices.length; i++)
        outp[i] = classNames[indices[i]];
    return outp
}

/*
set the table of the predictions
*/
function setTable(top5, probs) {
    //loop over the predictions
    for (let i = 0; i < top5.length; i++) {
        let sym = document.getElementById('sym' + (i + 1));
        let prob = document.getElementById('prob' + (i + 1));
        sym.innerHTML = top5[i];
        prob.innerHTML = Math.round(probs[i] * 100);
    }
    //create the pie
    createPie(".pieID.legend", ".pieID.pie");

}
