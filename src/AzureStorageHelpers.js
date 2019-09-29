import { Aborter, AnonymousCredential, BlockBlobURL, BlobURL, ContainerURL, StorageURL, ServiceURL} from "@azure/storage-blob";

export default class AzureStorageHelpers {
    static saveCarToBlobStorage(car, neuralnet, settings) {   
        if (car.body.fitness<settings.azureStorage.fitnessThreshold) {return;}
        let carToSave = {
            distance: car.body.distance,
            fitness: car.body.fitness,
            model: {
                layer1: {
                    weights: neuralnet.model.layers[0].getWeights()[0].dataSync(),
                    bias: neuralnet.model.layers[0].getWeights()[1].dataSync(),
                },
                layer2: {
                    weights: neuralnet.model.layers[1].getWeights()[0].dataSync(),
                    bias: neuralnet.model.layers[1].getWeights()[1].dataSync(),
                }
            }
        };

        let now = new Date();
        let carBlobName = `${now.getFullYear()}${now.getMonth()+1}${now.getDate()}_${now.getHours()}${now.getMinutes()}${now.getSeconds()}f${now.getMilliseconds()}_${carToSave.distance.laps}_${Math.round(carToSave.fitness)}.json`;

        const anonymousCredential = new AnonymousCredential();

        const pipeline = StorageURL.newPipeline(anonymousCredential);

        const serviceURL = new ServiceURL(
            settings.azureStorage.blobUri,
            pipeline
          );

        let containerURL = ContainerURL.fromServiceURL(serviceURL, 'models');
        let content = JSON.stringify(carToSave);

        const blobURL = BlobURL.fromContainerURL(containerURL, carBlobName);
        const blockBlobURL = BlockBlobURL.fromBlobURL(blobURL);
        const uploadBlobResponse = blockBlobURL.upload(Aborter.none, content, content.length)
                                        .then((data) => {console.log(data)})
                                        .catch((err) => {console.log(err)});
    }

    static loadFromBlobStorage(blobName, settings) {
        const anonymousCredential = new AnonymousCredential();

        const pipeline = StorageURL.newPipeline(anonymousCredential);

        const serviceURL = new ServiceURL(
            settings.azureStorage.blobUri,
            pipeline
          );

        let containerURL = ContainerURL.fromServiceURL(serviceURL, 'models');

        const blobURL = BlobURL.fromContainerURL(containerURL, blobName);

        return new Promise((resolve, reject) => {
            blobURL.download(Aborter.none, 0)
                .then((blob) => { 
                    blob.blobBody.then((body) => {
                        body.text()
                        .then((json) => { 
                            resolve(JSON.parse(json));
                        })
                        .catch((err) => { 
                            reject(err)
                        });
                    });
                })
                .catch((err) => reject(err));
        });
    }
}