export class SpriteButton extends Phaser.GameObjects.Sprite {
    constructor(scene, x, y, textures, callback) {
        super(scene, x, y, textures.out);

        this.scene = scene;

        this.setInteractive()
            .on('pointerdown', () => callback());
    }
};