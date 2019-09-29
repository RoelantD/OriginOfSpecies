class Button extends Phaser.GameObjects.Sprite {
    constructor(scene, x, y, textures, callback) {
        super(scene, x, y, textures.out);

        this.scene = scene;

        this.setInteractive()
            .on('pointerdown', () => callback());
    }

};


class Menu extends Phaser.Scene {
    constructor() {
        super({key: 'Menu'});
    }

    create() {
        this.add.image(2240, 1152, 'startBackground').setOrigin(1,1).setScale(0.7);

        this.add.image(1000, 500, 'startScreen').setOrigin(0.5, 0.5);

        this.add.existing(new Button(
            this, 960, 700,
            {
                out: 'startBtn'
            },
            () => { this.scene.start('Game') })
        );
    }
}

export { Menu };
