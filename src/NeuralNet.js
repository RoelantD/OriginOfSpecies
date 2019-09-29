import * as tf from '@tensorflow/tfjs';

export class NeuralNet {
    constructor(scene, generation, index, parents) {
        this.scene = scene;
        this.generation = generation;
        this.index = index;

        this.parents = (parents == null) ? Array(8).fill(generation.toString()+'_'+index.toString()) : parents;
        
        const NEURONS = 8;
        
        const hiddenLayer = tf.layers.dense({
            units: NEURONS,
            inputShape: [3],
            activation: 'sigmoid',
            kernelInitializer: 'leCunNormal',
            useBias: true,
            biasInitializer: 'randomNormal',        
        });
        
        const outputLayer = tf.layers.dense({
            units: 1, 
            activation: "sigmoid"
        });

        const model = tf.sequential();
        model.add(hiddenLayer);
        model.add(outputLayer);
        
        model.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });

        this.model = model;

        this.fitness = 0;
    }

    id() {
        return this.generation.toString()+'_'+this.index.toString();
    }

    updateHUD() {
        var weights = this.model.layers[0].getWeights()[0].dataSync();
        var bias = this.model.layers[0].getWeights()[1].dataSync();

        var neuralNetText = '';

        for (var hidden=0; hidden<8; hidden++) {
            neuralNetText = '';

            for (var inputNode=0; inputNode<3; inputNode++) {
                let dataId = (inputNode * 8) + hidden;
                neuralNetText = neuralNetText + weights[dataId].toFixed(2)+ "\n";
            }

            this.scene.neuralNetHud.hidden[hidden].setText(neuralNetText);
        }
    }
};