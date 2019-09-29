class Loader extends Phaser.Scene {
    constructor() {
        super({key: 'Loader'});
    }

    preload() {
        this.load.image('startBackground', 'assets/techorama_background.png');
        this.load.image('startScreen', 'assets/startscreen_techorama_8bit.png');

        this.load.image('trace', 'assets/trace.png');
        this.load.image('startBtn', 'assets/startBtn_techorama.png');
        this.load.image('startStop', 'assets/startstop.png');
        this.load.image('stopStart', 'assets/stopstart.png');
        this.load.image('skull', 'assets/skull.png');
        this.load.image('checkered', 'assets/checkered.png');
        this.load.image('flag', 'assets/win.png');
        this.load.image('play', 'assets/play.png');
        this.load.image('pause', 'assets/pause.png');

        this.load.image('sign', 'assets/techorama_road.png');
        this.load.image('car2', 'assets/car_red_small_5.png');
        this.load.image('car_big', 'assets/car_red_5.png');

        this.load.image('arrow_left', 'assets/arrow_left.png');
        this.load.image('arrow_top', 'assets/arrow_top.png');
        this.load.image('arrow_right', 'assets/arrow_right.png');
        this.load.image('steering', 'assets/steering.png');

        this.load.image('Racing_64', 'Racing-Tilemap_64.png');
        this.load.image('cross', 'assets/black_cross_small.png');
        
        this.load.tilemapTiledJSON('level1', 'Techorama_full_64.json');

        this.load.spritesheet('explosion', 'assets/explosion_small.png', { frameWidth: 40, frameHeight: 40 });
        this.load.spritesheet('playpause', 'assets/button_sprite.png', { frameWidth: 52, frameHeight: 52 });

        this.load.bitmapFont('8bit', 'assets/8bit.png', 'assets/8bit.fnt');
        this.load.bitmapFont('8bit_black', 'assets/8bit_black.png', 'assets/8bit_black.fnt');
    }

    create() {
        this.scene.start('Menu');
    }
}

export { Loader };
