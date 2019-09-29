
class GameOver extends Phaser.Scene {
    constructor() {
        super({key: 'GameOver'});
    }

    create () {
        this.add.image(600, 350, 'deathScreen');

        this.add.sprite(600, 520, 'startBtn')
            .setInteractive().on('pointerdown', () => {
                this.scene.start('Game');
            });
    }
}

export { GameOver };
