export default class Car {
    constructor(scene, x, y) {
        this.ANGLE_DELTA = 0.1;

        this.SPEED = 0.0;

        this.scene = scene;

        const kind = Phaser.Math.RND.pick(['car2']);

        let carImage = this.scene.matter.add.image(x, y, kind);
        carImage.setDepth(100);
        carImage.body.label = 'car';
        
        carImage.body.graphics = this.scene.add.graphics();
        carImage.body.graphics.clear();

        this.body = carImage;

        carImage.throttling = 0;

        carImage.fitness = 0;

        carImage.distance = {
            front: -1,
            left: -1,
            right: -1,
            total: 0,
            rotation: 0,
            laps: 0
        }

        carImage.isExploding = false;

        carImage.lastpos = {
            x: x,
            y: y
        }

        carImage.raycast = {
            front: {},
            left: {},
            right: {}
        }

        this.scene.collidingBodies = [];

        this.init(carImage, x, y);
    }

    destroy() {
        let carStatus = {
            x: this.body.x,
            y: this.body.y,
            speed: this.body.speed,
            fitness: this.body.fitness,
            raycast: this.body.raycast
        };

        this.body.setVisible(false);
        this.body.setActive(false);

        return carStatus;
    }

    init(obj, x, y) {
        obj
        .setMass(35)
        .setPosition(x, y)
        .setAngle(-90)
        .setScale(0.45)
        .setFriction(0.5, 0.3)
        .setFixedRotation()
        .setVelocity(0, 0);
    }

    reset(x, y) {
        this.init(this.body, x, y);

        this.body.setActive(true);
        this.body.setVisible(true);
    }

    rotate(delta) {
        this.body.setAngularVelocity(delta);
    }

    rotateClockwise() {
        this.rotate(this.ANGLE_DELTA);
    }

    rotateCounterclockwise() {
        this.rotate(-this.ANGLE_DELTA);
    }

    goForward() {
        this.body.thrust(this.SPEED);
    }

    goBackward() {
        this.body.thrust(-this.SPEED);
    }
};
