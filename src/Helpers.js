export default class Helpers {
    static getRootBody(body) {
        if (body.parent === body) {
            return body;
        }
        while (body.parent !== body) {
            body = body.parent;
        }
        return body;
    }

    static stopSidewaysVelocity(sprite) {
        if(!this.stopSidewaysVelocity.sideways) {
            this.stopSidewaysVelocity.sideways = {};
        }
    
        let body     = sprite.body;
        let velocity = body.velocity;
        let rotation = body.angle + Math.PI / 2;
        let sideways = this.stopSidewaysVelocity.sideways;
    
        sideways.x = Math.cos(rotation);
        sideways.y = Math.sin(rotation);
    
        let dot_product = velocity.x * sideways.x + velocity.y * sideways.y;
    
        velocity.x = sideways.x * dot_product;
        velocity.y = sideways.y * dot_product;
    }
    
    static shuffle(array) {
        let currentIndex = array.length;
        let temporaryValue, randomIndex;
    
        while (0 !== currentIndex) {
            randomIndex = Math.floor(Math.random() * currentIndex);
            currentIndex -= 1;

            temporaryValue = array[currentIndex];
            array[currentIndex] = array[randomIndex];
            array[randomIndex] = temporaryValue;
        }
    
        return array;
    
    };
}