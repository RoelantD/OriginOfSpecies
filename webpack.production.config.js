const path = require('path');
const CopyWebpackPlugin = require('copy-webpack-plugin');
const webpack = require('webpack');

module.exports = {
  entry: {
    app: './src/index.js',
    'vendor-libs': ['phaser']
  },

  output: {
    path: path.resolve(__dirname, 'dist'),
    filename: 'racing-car-game.js'
  },

  module: {
    rules: [
      {
        test: /\.js$/,
        include: path.resolve(__dirname, 'src/'),
        use: {
          loader: 'babel-loader',
          options: {
            presets: ['env']
          }
        }
      }
    ]
  },

  devServer: {
    contentBase: path.resolve(__dirname, 'dist'),
  },

  plugins: [
    new CopyWebpackPlugin([
      {
        from: path.resolve(__dirname, 'src', 'index.html'),
        to: path.resolve(__dirname, 'dist')
      },
      {
        from: path.resolve(__dirname, 'assets', '**', '*'),
        to: path.resolve(__dirname, 'dist')
      }
    ]),
    new webpack.optimize.UglifyJsPlugin({
      drop_console: true,
      minimize: true,
      output: {
        comments: false
      }
    }),
    new webpack.DefinePlugin({
      'typeof CANVAS_RENDERER': JSON.stringify(true),
      'typeof WEBGL_RENDERER': JSON.stringify(true)
    }),
    new webpack.optimize.CommonsChunkPlugin({
      name: 'vendor-libs',
      filename: 'vendor-libs.js'
    }),
  ],
}

