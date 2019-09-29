export default class Population {
    constructor(models, index) {
        this.models = models;
        this.index = index + 1;
    }

    statistics() {
        let stats = {
            averageFitness: this.calculateAverageFitness()

        };

        return stats;
    }

    calculateAverageFitness() {
        if (this.models.length) {
            let summed = this.models.reduce((total, b) => total + sanitize(b.fitness), 0);

            return  summed / this.models.length;
        } else {
            return 0;
        }
    }
};

function sanitize(input) {
    return (input) ? input: 0;
}