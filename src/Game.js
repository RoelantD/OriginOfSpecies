import * as tf from '@tensorflow/tfjs';
import Car from './Car';
import { SpriteButton } from './SpriteButton';
import { NeuralNet } from './NeuralNet';
import Population from './Population';
import Helpers from './Helpers';
import { ray as rayClass, vec2 } from './Raycast';
import AzureStorageHelpers from './AzureStorageHelpers';

class Game extends Phaser.Scene {
    constructor() {
        super({key: 'Game'});
    }

    //    ___             _               _           
    //   / __|___ _ _  __| |_ _ _ _  _ __| |_ ___ _ _ 
    //  | (__/ _ \ ' \(_-<  _| '_| || / _|  _/ _ \ '_|
    //   \___\___/_||_/__/\__|_|  \_,_\__|\__\___/_|  
    create() {
        this.map = this.add.tilemap('level1');

        this.loadTilesAndSetupCollisions();
        this.createSettings();
        this.createGlobalCollections();
        this.setupExplosions();
        this.setBounds();
        this.createInterface();
        this.createCar();
        this.createPopulation();
        this.createInput();
        this.createAnimations();
        this.createGenerationGraph();
        this.createTrackGraphics();
    }

    //  ___      _   _   _              
    // / __| ___| |_| |_(_)_ _  __ _ ___
    // \__ \/ -_)  _|  _| | ' \/ _` (_-<
    // |___/\___|\__|\__|_|_||_\__, /__/
    //                         |___/    
    createSettings() {
        this.settings = {
            populationSize: 5,
            currentCarIndex: 0,
            isRunning: false,
            rayCastOn: false,
            carsDied: 0,
            lapsFinished: 0,
            startingPoint: {x:192 ,y:578},
            measuringPoint: {x:540, y:450},
            azureStorage: {
                account: '[Azure blob storage account name]',
                blobUri: '[Blob service SAS URL]',
                saveToBlobStorage: true,
                fitnessThreshold: 2500
            },
            winningCarJson: '[Blob filename generated model]'
         };
    } 
 
    //   ___              _   _         _         __  __ 
    //  / __|___ _ _  ___| |_(_)__   __| |_ _  _ / _|/ _|
    // | (_ / -_) ' \/ -_)  _| / _| (_-<  _| || |  _|  _|
    //  \___\___|_||_\___|\__|_\__| /__/\__|\_,_|_| |_|  
    populationCycle() {
        if (this.settings.currentCarIndex === this.populationModels.length-1) {
            let addPopulation = new Population(this.populationModels, this.populations.length);
            
            this.populations.push(addPopulation);

            this.refreshPopulationGraph();
            this.mutatePopulation(addPopulation);
            this.cleanupTrack();
        } 

        this.selectNextCar();
        this.createCar();
    }

    mutatePopulation(population) {
        this.populationModels = [];

        let parents = this.getTop(5, population);                           // 5
        let crossedOver = this.crossOver(parents);                          // 10
        let randomSubset = this.randomSet(crossedOver, 5); 
        let mutatedCrossedOver =  this.mutate(randomSubset);                // 5
        let mutated = this.mutate(parents);                                 // 5
        let immigration = [new NeuralNet(this, 0, 0),new NeuralNet(this, 0, 0),new NeuralNet(this, 0, 0),new NeuralNet(this, 0, 0),new NeuralNet(this, 0, 0)]; // 5

        this.populationModels=this.populationModels.concat(parents);
        this.populationModels=this.populationModels.concat(crossedOver);
        this.populationModels=this.populationModels.concat(mutatedCrossedOver);
        this.populationModels=this.populationModels.concat(mutated);
        this.populationModels=this.populationModels.concat(immigration);

        this.populationModels.map((m, idx) => { m.index = idx; m.generation = this.populations.length; m.fitness = 0; });

        Helpers.shuffle(this.populationModels);

        this.settings.currentCarIndex = 0;
        this.setNextCar();
    }

    crossOver(parents) {
        let totalNumberOfNeurons = 8;
        let children = [];
        
        for (let i = 0; i < parents.length; i++){
            for (let j = i + 1; j < parents.length; j++){
                let parent1 = parents[i];
                let parent2 = parents[j];

                let startCrossOver = Math.floor(Math.random() * totalNumberOfNeurons);  // Number 0-7
                let maxNumberOfNeurons = (totalNumberOfNeurons - startCrossOver);       // Never take all neurons -> 7
                let numberOfNeurons = Math.ceil(Math.random() * maxNumberOfNeurons);    // Number 1-7
                let endCrossOver = startCrossOver + numberOfNeurons;

                let child = new NeuralNet(parent1.scene, parent1.generation, parent1.index);
                
                this.crossOverHiddenLayer(child, parent1, parent2, startCrossOver, endCrossOver);
                this.crossOverOutputLayer(child, parent1, parent2, startCrossOver, endCrossOver);

                children.push(child);
            }
        }

        return children;
    }
    
    crossOverHiddenLayer(child, parent1, parent2, startCrossOver, endCrossOver) {
        let ancestory = Array.from(parent1.parents);

        let modelData1 = parent1.model.layers[0].getWeights()      
        let modelData2 = parent2.model.layers[0].getWeights()      

        let weights1 = Object.assign(new Float32Array(24), modelData1[0].dataSync());
        let bias1 = Object.assign(new Float32Array(8), modelData1[1].dataSync());

        let weights2 = Object.assign(new Float32Array(24), modelData2[0].dataSync());
        let bias2 = Object.assign(new Float32Array(8), modelData2[1].dataSync());

        let weightsChild = new Float32Array(weights1);
        let biasChild = new Float32Array(bias1);

        for (let inputNode=0; inputNode<3; inputNode++) {
            for (let hidden1=startCrossOver; hidden1<endCrossOver; hidden1++) {
                let dataId = (inputNode * 8) + hidden1;
                
                weightsChild[dataId] = weights2[dataId];
            }
        }
        
        for (let dataId=startCrossOver; dataId<endCrossOver; dataId++) {
            biasChild[dataId] = bias2[dataId];
            ancestory[dataId] = parent2.parents[dataId];
        }
                        
        let weightsShape = modelData1[0].shape;
        let biasShape = modelData1[1].shape;

        child.model.layers[0].setWeights([tf.tensor(weightsChild, weightsShape), tf.tensor(biasChild, biasShape)]);

        child.parents = ancestory;
    }

    crossOverOutputLayer(child, parent1, parent2, startCrossOver, endCrossOver) {
        let modelData1 = parent1.model.layers[1].getWeights()      
        let modelData2 = parent2.model.layers[1].getWeights()      

        let weights1 = Object.assign(new Float32Array(8), modelData1[0].dataSync());
        let bias1 = Object.assign(new Float32Array(1), modelData1[1].dataSync());

        let weights2 = Object.assign(new Float32Array(8), modelData2[0].dataSync());
        let bias2 = Object.assign(new Float32Array(1), modelData2[1].dataSync());

        let weightsChild = new Float32Array(weights1);
        let biasChild = new Float32Array(bias1);

        for (let outputNode=startCrossOver; outputNode<endCrossOver; outputNode++) {
            weightsChild[outputNode] = weights2[outputNode];
        }
        
        for (let dataId=startCrossOver; dataId<endCrossOver; dataId++) {
            biasChild[dataId] = bias2[dataId];
        }
                        
        let weightsShape = modelData1[0].shape;
        let biasShape = modelData1[1].shape;

        child.model.layers[1].setWeights([tf.tensor(weightsChild, weightsShape), tf.tensor(biasChild, biasShape)]);
    }

    mutate(parents) {
        let totalNumberOfNeurons = 8;
        let children = [];
        
        for (let i = 0; i < parents.length; i++){
            let parent = parents[i];

            let child = Object.assign(new NeuralNet(parent.scene, parent.generation, parent.index), parent);

            this.mutateLayer(parent.model.layers[0], child.model.layers[0], 0.15)
            this.mutateLayer(parent.model.layers[1], child.model.layers[1], 0.15)
            
            children.push(child);
        }

        return children;
    }
    
    mutateLayer(parentLayer, childLayer, chanceOnMutation) {
        let modelData1 = parentLayer.getWeights()      

        let weightData = modelData1[0].dataSync();
        let biasData = modelData1[1].dataSync();
        let totalNumberOfNeurons = biasData.length;

        let weights = Object.assign(new Float32Array(weightData.length), weightData);
        let bias = Object.assign(new Float32Array(biasData.length), biasData);

        for (let inputNode=0; inputNode<3; inputNode++) {
            for (let neuronIndex=0; neuronIndex<totalNumberOfNeurons; neuronIndex++) {
                let dataId = (inputNode * 8) + neuronIndex;

                // 15% chance of mutation
                let chance = Math.random();

                if (chance <=chanceOnMutation) {
                    let mutationFactor = (0.5 - Math.random()); // Max variance = -0.5 / 0.5
                    weights[dataId] = (weights[dataId] + mutationFactor);
                }
            }
        }

        for (let neuronIndex=0; neuronIndex<totalNumberOfNeurons; neuronIndex++) {
            let dataId = neuronIndex;

            // 15% chance of mutation
            let chance = Math.random();

            if (chance <=chanceOnMutation) {
                let mutationFactor = (0.5 - Math.random()); // Max variance = -0.5 / 0.5
                bias[dataId] = (bias[dataId] + mutationFactor);
            }
        }

        let weightsShape = modelData1[0].shape;
        let biasShape = modelData1[1].shape;

        childLayer.setWeights([tf.tensor(weights, weightsShape), tf.tensor(bias, biasShape)]);
    }

    randomSet(parents, numberOfItems) {
        let children = [];

        let numberOfModelsToGet = Math.min(numberOfItems, parents.length);

        if (numberOfModelsToGet === 0 ) {
            return [];
        }

        for (let childIndex = 0; childIndex < numberOfModelsToGet; childIndex++) {
            let randomIndex = Math.floor(Math.random() * parents.length);

            let parent = parents[randomIndex];

            let child = Object.assign(new NeuralNet(parent.scene, parent.generation, parent.index), parent);
            children.push(child);
        }

        return children;
    }

    getTop(numberOfModels, population) {
        let sortedModels = population.models.sort((a, b) => a.fitness > b.fitness ? -1 : 1);

        let numberOfModelsToGet = Math.min(numberOfModels, sortedModels.length);

        if (numberOfModelsToGet === 0 ) {
            return [];
        }

        let children = sortedModels.slice(0, numberOfModelsToGet);

        return children;
    }

    //  ___ _                                  _        _               
    // | _ \ |_  __ _ ___ ___ _ _   _ __  __ _(_)_ _   | |___  ___ _ __ 
    // |  _/ ' \/ _` (_-</ -_) '_| | '  \/ _` | | ' \  | / _ \/ _ \ '_ \
    // |_| |_||_\__,_/__/\___|_|   |_|_|_\__,_|_|_||_| |_\___/\___/ .__/
    //                                                            |_|   
    update(time, delta) {
        if (!this.settings.isRunning) { 
            this.toggleRunningState.setFrame(1);
            return; 
        } else {
            this.toggleRunningState.setFrame(0);
        }

        if (this.neuralnet == null) { return; }
        if (!this.car || !this.car.body || !this.car.body.visible) { return; }

        // Init
        let car = this.car.body;
        
        // Make prediction
        let tens = tf.tensor([[car.distance.left, car.distance.front, car.distance.right]]);
        let result = this.neuralnet.model.predict(tens);
        let prediction = result.dataSync()[0];
        this.neuralNetHud.output.setText(prediction.toFixed(2));

        // Turn car according to prediction
        if (prediction < 0.48) {
            this.car.rotateClockwise();
        } else if (prediction > 0.52) {
            this.car.rotateCounterclockwise();
        }

        // Move car
        car.thrust(0.25);

        Helpers.stopSidewaysVelocity(car);

        // Calculate distances from car to relevant walls
        this.calculateCollisionDistances(car);

        // Calculate degrees in radians travelled from fixed space in middle map
        this.setFitnessParameters(car);

        // Update car parameters for next round
        this.setCarParameters(car);

        // Update HUD data
        this.updateInputs(car.distance);
        this.updateCarHud(car);

        // Finish HIM
        this.checkForFinishCondition(car)
    }


















    //  __  __        _   _                                 _        _   _               _         __  __ 
    // |  \/  |___ __| |_| |_  _   _ __ _ _ ___ ___ ___ _ _| |_ __ _| |_(_)___ _ _    __| |_ _  _ / _|/ _|
    // | |\/| / _ (_-<  _| | || | | '_ \ '_/ -_|_-</ -_) ' \  _/ _` |  _| / _ \ ' \  (_-<  _| || |  _|  _|
    // |_|  |_\___/__/\__|_|\_, | | .__/_| \___/__/\___|_||_\__\__,_|\__|_\___/_||_| /__/\__|\_,_|_| |_|  
    //                      |__/  |_|                                                                     
    loadTilesAndSetupCollisions() {
        let tileset = this.map.addTilesetImage('Racing_64');
        let layer1 = this.map.createDynamicLayer('Tile Layer 1' , tileset, 0, 0);
        let layer3 = this.map.createDynamicLayer('Tile Layer 3' , tileset, 0, 0);
        let layer4 = this.map.createDynamicLayer('Tile Layer 4' , tileset, 0, 0);
        let layer5 = this.map.createDynamicLayer('Tile Layer 5' , tileset, 0, 0);
        let collisionLayer = this.map.createDynamicLayer('Tile Layer 2' , tileset, 0, 0);
        collisionLayer.setCollisionFromCollisionGroup();
        this.matter.world.convertTilemapLayer(collisionLayer);
    }

    createGlobalCollections() {
        this.populations = [];
        this.parents = [];
    }

    createTrackGraphics() {
        this.add.sprite(303, 800, 'sign').setScale(0.4).setRotation(-0.45).setAlpha(0.5);
    }

    setupExplosions() {
        this.explosions = this.add.group();
        this.matter.world.on('collisionstart', function (event) {
            for (let i = 0; i < event.pairs.length; i++) {
                let bodyA = Helpers.getRootBody(event.pairs[i].bodyA);
                let bodyB = Helpers.getRootBody(event.pairs[i].bodyB);
                
                if (bodyB.gameObject.visible) {
                    this.explodeCar();
                }
            }
        }, this);
    }

    createGenerationGraph() {
        this.graphGraphicsSettings = {
            startX: 1555,
            startY: 320,
            width: 400,
            height: 180,
            maxGeneration: 10,
            heightLines: 4
        };

        this.graphGraphics = this.add.graphics({ x: this.graphGraphicsSettings.startX, y: this.graphGraphicsSettings.startY  });
        this.graphGraphics.fillStyle(0xcccccc, 1);
        this.graphGraphics.lineStyle(2, 0x888888);
        this.graphGraphics.fillRect(0, 0, this.graphGraphicsSettings.width, this.graphGraphicsSettings.height);
        this.graphGraphics.lineStyle(1, 0xAAAAAA, 0.5);

        this.graphGraphicsLines = this.add.graphics({ x: this.graphGraphicsSettings.startX, y: this.graphGraphicsSettings.startY  });

        for (let hor = 0; hor<this.graphGraphicsSettings.heightLines; hor++) {
            this.graphGraphicsLines.lineBetween(0,hor*(this.graphGraphicsSettings.height / this.graphGraphicsSettings.heightLines),this.graphGraphicsSettings.width,hor*(this.graphGraphicsSettings.height / this.graphGraphicsSettings.heightLines));
        }

        for (let ver = 0; ver<this.graphGraphicsSettings.maxGeneration; ver++) {
            this.graphGraphicsLines.lineBetween(ver*(this.graphGraphicsSettings.width / this.graphGraphicsSettings.maxGeneration),this.graphGraphicsSettings.height,ver*(this.graphGraphicsSettings.width / this.graphGraphicsSettings.maxGeneration),0);
        }

        this.graphMaxFitness = this.add.bitmapText(1480,  this.graphGraphicsSettings.startY, "8bit", '500', 18, 0);
        this.graphHalfFitness = this.add.bitmapText(1480,  this.graphGraphicsSettings.startY + (this.graphGraphicsSettings.height / 2), "8bit", '250', 18, 0);

        this.add.bitmapText(this.graphGraphicsSettings.startX,  this.graphGraphicsSettings.startY -30, "8bit", 'Evolution graph', 20);

        this.add.bitmapText(this.graphGraphicsSettings.startX - 14,  this.graphGraphicsSettings.startY + this.graphGraphicsSettings.height + 5, "8bit", '0', 18);
        this.graphMaxFitness.setX( this.graphGraphicsSettings.startX - 4 - this.graphMaxFitness.width);
        this.graphHalfFitness.setX( this.graphGraphicsSettings.startX - 4 - this.graphHalfFitness.width);

        this.graphHalfGenerations = this.add.bitmapText( this.graphGraphicsSettings.startX + (this.graphGraphicsSettings.width / 2) - 15,  this.graphGraphicsSettings.startY + this.graphGraphicsSettings.height + 5, "8bit", (this.graphGraphicsSettings.maxGeneration/2), 18, 0);
        this.graphMaxGenerations = this.add.bitmapText( this.graphGraphicsSettings.startX + this.graphGraphicsSettings.width - 30,  this.graphGraphicsSettings.startY + this.graphGraphicsSettings.height + 5, "8bit", this.graphGraphicsSettings.maxGeneration, 18, 0);
        
        this.graphHalfGenerations.setX( this.graphGraphicsSettings.startX + (this.graphGraphicsSettings.width / 2) - (this.graphHalfGenerations.width / 2));

        this.graphGraphicData = this.add.graphics({ x: this.graphGraphicsSettings.startX, y: this.graphGraphicsSettings.startY  });
        this.graphGraphicData.lineStyle(2, 0xFF0000, 1);
    }

    createCarDashboard() {
        this.carDashGraphics = this.add.graphics();
        this.carDashGraphics.fillStyle(0xcccccc, 1);

        let inputCenters = [[1550,680],[1690,680],[1830,680]];
        let outputCenters = [[1690,1070]];
        let hiddenCenters = [[1475,770],[1475,850],[1475,930],[1475,1010],
                            [1905,770],[1905,850],[1905,930],[1905,1010]];

        this.carDashGraphics.lineStyle(2, 0x888888);

        for (let inputIndex = 0; inputIndex<inputCenters.length; inputIndex++) {
            for (let hiddenIndex = 0; hiddenIndex<hiddenCenters.length; hiddenIndex++) {
                this.carDashGraphics.strokeLineShape(new Phaser.Geom.Line(inputCenters[inputIndex][0], inputCenters[inputIndex][1], hiddenCenters[hiddenIndex][0], hiddenCenters[hiddenIndex][1])); 
            } 
        }

        for (let outputIndex = 0; outputIndex<outputCenters.length; outputIndex++) {
            for (let hiddenIndex = 0; hiddenIndex<hiddenCenters.length; hiddenIndex++) {
                this.carDashGraphics.strokeLineShape(new Phaser.Geom.Line(outputCenters[outputIndex][0], outputCenters[outputIndex][1], hiddenCenters[hiddenIndex][0], hiddenCenters[hiddenIndex][1])); 
            } 
        }

        //Input
        this.carDashGraphics.fillRoundedRect(1500, 650, 100, 60, 14);
        this.carDashGraphics.fillRoundedRect(1640, 650, 100, 60, 14);
        this.carDashGraphics.fillRoundedRect(1780, 650, 100, 60, 14);

        //Output
        this.carDashGraphics.fillRoundedRect(1640, 1040, 100, 60, 14);

        //Hidden
        this.carDashGraphics.fillRoundedRect(1425, 740, 100, 60, 14);
        this.carDashGraphics.fillRoundedRect(1425, 820, 100, 60, 14);
        this.carDashGraphics.fillRoundedRect(1425, 900, 100, 60, 14);
        this.carDashGraphics.fillRoundedRect(1425, 980, 100, 60, 14);

        this.carDashGraphics.fillRoundedRect(1855, 740, 100, 60, 14);
        this.carDashGraphics.fillRoundedRect(1855, 820, 100, 60, 14);
        this.carDashGraphics.fillRoundedRect(1855, 900, 100, 60, 14);
        this.carDashGraphics.fillRoundedRect(1855, 980, 100, 60, 14);
        
        let arrowLeft = this.add.sprite(1530, 630, 'arrow_left').setScale(1.6);
        let arrowTop = this.add.sprite(1690, 630, 'arrow_top').setScale(1.7);
        let arrowRight = this.add.sprite(1850, 630, 'arrow_right').setScale(1.6);
        let steering = this.add.sprite(1690, 1020, 'steering').setScale(1.3);
        let largeCarSprite = this.add.sprite(1690, 860, 'car_big').setScale(2.3);

        this.neuralNetHud = {
            input: [
                this.add.bitmapText(1520, 670, "8bit_black", '888', 18, 0),
                this.add.bitmapText(1661, 670, "8bit_black", '888', 18, 0),
                this.add.bitmapText(1800, 670, "8bit_black", '888', 18, 0),
            ],
            hidden: [
                this.add.bitmapText(1448, 748, "8bit_black", '0.88\n0.88\n0.88', 12, 0),
                this.add.bitmapText(1448, 828, "8bit_black", '0.88\n0.88\n0.88', 12, 0),
                this.add.bitmapText(1448, 908, "8bit_black", '0.88\n0.88\n0.88', 12, 0),
                this.add.bitmapText(1448, 988, "8bit_black", '0.88\n0.88\n0.88', 12, 0),
                this.add.bitmapText(1881, 748, "8bit_black", '0.88\n0.88\n0.88', 12, 0),
                this.add.bitmapText(1881, 828, "8bit_black", '0.88\n0.88\n0.88', 12, 0),
                this.add.bitmapText(1881, 908, "8bit_black", '0.88\n0.88\n0.88', 12, 0),
                this.add.bitmapText(1881, 988, "8bit_black", '0.88\n0.88\n0.88', 12, 0)
            ],
            output: this.add.bitmapText(1655, 1060, "8bit_black", '888', 18, 0)
        };

        let font = { font: "38px atari", fill: "#ff0044", align: "right" };
        
        this.topList = this.add.text(450, 1900, "", font);

        this.populationHUD = this.add.graphics(500, 1400);
    }

    createDataDashboard() {
        const legend =  "Generation:\n" +
                        "Car index:\n" +
                        "Current fitness:\n" +
                        "Previous fitness:\n" +
                        "Cars died:\n" +
                        "Laps finished:\n";

        this.add.sprite(1830, 205, 'skull').setScale(0.9);

        this.hudLegend = this.add.bitmapText(1410, 60, "8bit", legend, 24, 0); 
        this.hudLegend.lineSpacing = 100;

        const dummyData =  "1\n" +
                           "88888\n" +
                           "88888\n" +
                           "88888";

        this.hud = this.add.bitmapText(1850, 60, "8bit", dummyData, 24, 0); 
    }

    createAnimations() {
        this.anims.create({
            key: 'explode',
            frames: this.anims.generateFrameNumbers('explosion'),
            frameRate: 24,
            repeat: 0
        });
    }

    refreshPopulationGraph() {
        const allFitnessValues = this.populations.map(population => population.statistics().averageFitness);

        let maxFitness = Math.max(...allFitnessValues);
        let maxGraphFitness = Math.ceil(maxFitness / 100) * 100;

        this.graphMaxFitness.setText(maxGraphFitness);
        this.graphHalfFitness.setText(maxGraphFitness / 2);
        this.graphMaxFitness.setX(this.graphGraphicsSettings.startX - 4 - this.graphMaxFitness.width);
        this.graphHalfFitness.setX(this.graphGraphicsSettings.startX - 4 - this.graphHalfFitness.width);

        if (this.populations.length>10) {
            this.graphGraphicsLines.clear();
            this.graphHalfGenerations.setText('');
            this.graphMaxGenerations.setText(this.populations.length);
        }

        this.graphGraphicData.clear();
        this.graphGraphicData.lineStyle(3, 0xFF0000, 1);

        let lastPoint ={x:0, y:this.graphGraphicsSettings.height};

        this.populations.forEach((population, index) => {
            let newPoint = {
                x: Math.round((index + 1) * (this.graphGraphicsSettings.width / Math.max(this.graphGraphicsSettings.maxGeneration, this.populations.length))),
                y: this.graphGraphicsSettings.height - Math.round((population.statistics().averageFitness / maxGraphFitness) * this.graphGraphicsSettings.height)
            }

            this.graphGraphicData.lineBetween(lastPoint.x, lastPoint.y, newPoint.x, newPoint.y);

            lastPoint = newPoint;
        }, this);
    }

    cleanupTrack() {
        let explosions = this.explosions.getChildren();
        let fittestModels = explosions.sort((a, b) => a.fitness > b.fitness ? -1 : 1);

        let fittestModel = fittestModels[0];

        if (!this.blackCross) {
            this.blackCross = this.add.sprite(fittestModel.x, fittestModel.y, 'cross');
            this.blackCross.setDepth(2);
            this.blackCross.fitness = fittestModel.fitness;
        } else {
            if (fittestModel.fitness > this.blackCross.fitness) {
                this.blackCross.setX(fittestModel.x).setY(fittestModel.y);
                this.blackCross.fitness = fittestModel.fitness;
            }
        }
                
        for (let i=explosions.length-1; i>=0; i--) {
            explosions[i].destroy();
        }

        this.explosions.clear();
    }

    setBounds() {
        this.matter.world.setBounds(0, 0, 2560, 2560);
    }

    createPopulation() {
        this.populationModels = [];

        for (let carIndex=0; carIndex<this.settings.populationSize; carIndex++){
            this.populationModels[carIndex] = new NeuralNet(this, 0, carIndex);
        }

        this.settings.currentCarIndex = 0;
        this.setNextCar();
    }

    selectNextCar() {
        this.settings.carsDied++;

        if (this.settings.azureStorage.saveToBlobStorage) { AzureStorageHelpers.saveCarToBlobStorage(this.car, this.neuralnet, this.settings); }

        if (this.settings.currentCarIndex + 1 < this.populationModels.length) {
            this.settings.currentCarIndex++;
            this.setNextCar();
        } else {
            this.neuralnet = null;
        }
    }

    resetCar() {
        this.car.body.distance = {
            front: -1,
            left: -1,
            right: -1,
            total: 0,
            rotation: 0,
            laps: 0
        }
    }

    setNextCar() {
        this.resetCar();

        this.neuralnet = this.populationModels[this.settings.currentCarIndex];
        this.neuralnet.updateHUD();
    }

    saveNeuralNetState(carFitness) {
        this.populationModels[this.settings.currentCarIndex].fitness = carFitness;
    }

    explodeCar(hasLapped = false) {
        let carStatus = this.car.destroy();

        this.saveNeuralNetState(carStatus.fitness);

        this.drawExplosion(carStatus.x, carStatus.y, carStatus.fitness, hasLapped);
    }

    drawExplosion(carX, carY, carFitness, hasLapped) {
        if (hasLapped) {
            this.populationCycle();
        } else {
            let explosionSprite = this.add.sprite(carX, carY, 'explosion');
            explosionSprite.enable = false;
            explosionSprite.fitness = carFitness;
    
            explosionSprite.setDepth(1);
            this.explosions.add(explosionSprite);
            explosionSprite.play('explode');
    
            explosionSprite.on('animationcomplete', this.populationCycle, this);
        }
    }

    createCar() {
        if (this.car) {
            this.car.reset(this.settings.startingPoint.x, this.settings.startingPoint.y);
        } else {
            this.car = new Car(this, this.settings.startingPoint.x, this.settings.startingPoint.y);
        }
    }
    
    createInterface() {
        this.createCarDashboard();
        this.createDataDashboard();
    }

    createInput() {
        this.cursors = this.input.keyboard.createCursorKeys();

        this.counterclockwise = this.input.keyboard.addKey(Phaser.Input.Keyboard.KeyCodes.A);
        this.clockwise = this.input.keyboard.addKey(Phaser.Input.Keyboard.KeyCodes.D);
        this.forward = this.input.keyboard.addKey(Phaser.Input.Keyboard.KeyCodes.W);
        this.backward = this.input.keyboard.addKey(Phaser.Input.Keyboard.KeyCodes.S);

        [
            this.cursors.up, this.cursors.down, this.cursors.left,
            this.cursors.right, this.counterclockwise, this.clockwise,
            this.forward, this.backward,
        ].forEach(key => {
            key.reset();
        });

        this.spaceBar = this.input.keyboard.addKey(Phaser.Input.Keyboard.KeyCodes.SPACE);

        this.spaceBar.on('up', function (event) { this.settings.isRunning = !this.settings.isRunning; }, this);

        this.toggleRunningState = this.add.existing(new SpriteButton(
            this, 1420, 360,
            {
                out: 'playpause'
            },
            () => { this.settings.isRunning = !this.settings.isRunning; })
        ).setScale(1.1);

        this.add.existing(new SpriteButton(
            this, 1420, 430,
            {
                out: 'flag'
            },
            () => { this.loadWinningCarFromBlobStorage(); }).setScale(1.1)
        );

        this.add.existing(new SpriteButton(
            this, 1420, 500,
            {
                out: 'trace'
            },
            () => { this.settings.rayCastOn = !this.settings.rayCastOn; console.log(this.settings.rayCastOn); }).setScale(1.1)
        );
    }

    loadWinningCarFromBlobStorage() {
        let carBlobName = this.settings.winningCarJson;

        AzureStorageHelpers.loadFromBlobStorage(carBlobName, this.settings)
            .then((carToLoad) => {
                this.settings.isRunning = false;
                        
                this.car.reset(this.settings.startingPoint.x, this.settings.startingPoint.y);
                this.resetCar()
    
                this.updateRunningNeuralNet(carToLoad);
            })
            .catch((err) => console.log(err));
    }

    updateRunningNeuralNet(json) {
        this.neuralnet.model.layers[0].setWeights([tf.tensor(this.parse(json.model.layer1.weights), [3,8]), tf.tensor(this.parse(json.model.layer1.bias), [8])]);
        this.neuralnet.model.layers[1].setWeights([tf.tensor(this.parse(json.model.layer2.weights), [8,1]), tf.tensor(this.parse(json.model.layer2.bias), [1])]);

        this.settings.isRunning = true;
    }

    parse(data) {
        let arrayedObject = Object.values(data);
        return Object.assign(new Float32Array(arrayedObject.length), arrayedObject);
    }

    updateCarHud(car) {
        this.hud.setText((this.populations.length+1) + "\n" + 
        this.settings.currentCarIndex+ "/" + this.populationModels.length + "\n" + 
        Math.round(car.fitness) + "\n" + 
        Math.round(((this.settings.currentCarIndex>0) ?this.populationModels[this.settings.currentCarIndex-1].fitness : 0)) + "\n" +
        (this.settings.carsDied) + "\n" +
        (this.settings.lapsFinished)
        );
    }

    checkForFinishCondition(car) {
        if (car.distance.laps >= 1) {
            this.explodeCar(true);

            this.blackCross.destroy();
            this.blackCross = this.add.sprite(car.x, car.y, 'checkered').setAlpha(0.7);
            this.blackCross.fitness = car.fitness;

            this.settings.lapsFinished++;
            if (this.settings.lapsFinished > 0) {
                if (!this.raceFlag) {
                    this.raceFlag = this.add.sprite(1827, 238, 'checkered').setScale(0.9);
                }
            }
        }
    }

    setCarParameters(car) {
        car.fitness = this.calculateFitness();
        car.distance.total += Phaser.Math.Distance.Between(car.lastpos.x, car.lastpos.y, car.x, car.y);
        car.lastpos = {
            x: car.x,
            y: car.y
        };
    }

    setFitnessParameters(car) {
        let angleRadians = this.calcAngle(this.settings.startingPoint.x, this.settings.startingPoint.y+6, this.settings.measuringPoint.x, this.settings.measuringPoint.y, car.x, car.y, this.settings.measuringPoint.x, this.settings.measuringPoint.y);

        let roundedRadiansNew = Math.round(angleRadians);
        let roundedRadiansOld = Math.round(car.distance.rotation);

        if (roundedRadiansNew === 6 && roundedRadiansOld <= 1) {
            car.distance.laps = car.distance.laps - 1;
        } else if ( roundedRadiansOld === 6 && roundedRadiansNew <= 1) {
            car.distance.laps = car.distance.laps + 1;
        }

        car.distance.rotation = angleRadians;
    }

    calculateCollisionDistances(car) {
        let curX = car.x;
        let curY = car.y;
        let curA = car.angle;

        if (curA<0) {
            curA=-1*curA;
        } else {
            curA=360-curA;
        }

        let curRF = (curA - 90) * Math.PI / 180;
        let curRR = (curA - 135) * Math.PI / 180;
        let curRL = (curA - 45) * Math.PI / 180;

        let frXT =  curX - (Math.sin(curRF) * 18);
        let frYT =  curY - (Math.cos(curRF) * 18);

        let toXT =  curX - (Math.sin(curRF) * 1000);
        let toYT =  curY - (Math.cos(curRF) * 1000);

        let frXL =  curX - (Math.sin(curRL) * 18);
        let frYL =  curY - (Math.cos(curRL) * 18);
        
        let toXL =  curX - (Math.sin(curRL) * 600);
        let toYL =  curY - (Math.cos(curRL) * 600);

        let frXR =  curX - (Math.sin(curRR) * 18);
        let frYR =  curY - (Math.cos(curRR) * 18);
        
        let toXR =  curX - (Math.sin(curRR) * 600);
        let toYR =  curY - (Math.cos(curRR) * 600);

        this.calculateCollision(car, {x: frXT, y: frYT}, {x: toXT, y: toYT}, 1000, "front");
        this.calculateCollision(car, {x: frXL, y: frYL}, {x: toXL, y: toYL}, 1000, "left");
        this.calculateCollision(car, {x: frXR, y: frYR}, {x: toXR, y: toYR}, 1000, "right");

        car.body.graphics.clear();
        
        if (this.settings.rayCastOn) {
            car.body.graphics.lineStyle(3, 0x0000FF, 0.5); 

            let measureCenter = new Phaser.Geom.Circle(this.settings.measuringPoint.x, this.settings.measuringPoint.y, 3);
            // let startPoint = new Phaser.Geom.Circle(this.settings.startingPoint.x, this.settings.startingPoint.y, 3);
            car.body.graphics.strokeCircleShape(measureCenter);
            // car.body.graphics.strokeCircleShape(startPoint);

            car.body.graphics.strokeLineShape(new Phaser.Geom.Line(frXT, frYT, toXT, toYT)); 
            car.body.graphics.strokeLineShape(new Phaser.Geom.Line(frXL, frYL, toXL, toYL)); 
            car.body.graphics.strokeLineShape(new Phaser.Geom.Line(frXR, frYR, toXR, toYR)); 

            let corc = new Phaser.Geom.Circle(car.raycast.front.x, car.raycast.front.y, 5);
            let corcL = new Phaser.Geom.Circle(car.raycast.left.x, car.raycast.left.y, 5);
            let corcR = new Phaser.Geom.Circle(car.raycast.right.x, car.raycast.right.y, 5);
    
            car.body.graphics.strokeCircleShape(corc);
            car.body.graphics.strokeCircleShape(corcL);
            car.body.graphics.strokeCircleShape(corcR);
        }
    }

    updateInputs(distance) {
        this.neuralNetHud.input[0].setText(Math.round((distance.left>0)?distance.left:0));
        this.neuralNetHud.input[1].setText(Math.round((distance.front>0)?distance.front:0));
        this.neuralNetHud.input[2].setText(Math.round((distance.right>0)?distance.right:0));
    }

    calculateFitness() {
        let wrongDirection = (this.car.body.distance.laps < 0);
        return (((wrongDirection) ? -1 : 1) * this.car.body.distance.rotation * 1000) + (this.car.body.distance.laps * 10000);
    }

    calcAngle (x00,y00,x01,y01,x10,y10,x11,y11) {
        let dx0  = x01 - x00;
        let dy0  = y01 - y00;
        let dx1  = x11 - x10;
        let dy1  = y11 - y10;
        let angle = Math.atan2(dx0 * dy1 - dx1 * dy0, dx0 * dx1 + dy0 * dy1);

        if (angle < 0) {
            angle = (Math.PI*2)+angle;
        }

        return angle;
    }

    calculateCollision(child, start, end, dist, target) {
        let collisions = this.raycast(this.matter.world.localWorld.bodies, start, end, dist );

         if (collisions && collisions.body) {
            if (!this.collidingBodies.some((e) => e.id === collisions.body.id) ) {
                if (this.collidingBodies.length >= 10) { this.collidingBodies.shift(); }
                this.collidingBodies.push(collisions.body);
            }

            if (collisions && collisions.point) {
                let subtract = (target === 'front') ? 27 : 28;

                child.distance[target] = (Math.sqrt((Math.pow(collisions.point.x - child.x, 2) + Math.pow(collisions.point.y - child.y, 2)))) - subtract;
                child.raycast[target] = collisions.point;
            }
        }
    }

    raycast(bodies, start, r, dist){
        let targetVector = {x: r.x-start.x, y:r.y-start.y};

        let normRay = Phaser.Physics.Matter.Matter.Vector.normalise(targetVector);
        let ray = normRay;

        let collidingBodies2 =Phaser.Physics.Matter.Matter.Query.ray(bodies, start, r, 2); 
        if (collidingBodies2.length) {
            let collidingRay = new rayClass(vec2.fromOther(start),vec2.fromOther(r));

            let bodCol = [];

            for(let bodyIndex in collidingBodies2) {
                bodCol = bodCol.concat(rayClass.bodyCollisions(collidingRay, collidingBodies2[bodyIndex].body));
            }


            bodCol.sort(function(a,b){
                return a.point.distance(start) - b.point.distance(start);
            });

            if (!bodCol.length) {
                for(let bodyIndex in collidingBodies2) {
                    bodCol = bodCol.concat(rayClass.bodyCollisions(collidingRay, collidingBodies2[bodyIndex].body));
                }
    
                bodCol.sort(function(a,b){
                    return a.point.distance(start) - b.point.distance(start);
                });
            }

            return {point: bodCol[0].point, body: bodCol[0].body}; 
        }
        
        return;
    }
}

export { Game };
