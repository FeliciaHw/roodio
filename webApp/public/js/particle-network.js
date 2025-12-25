// Set up ParticleNetwork appropriately for the environment.
(function (factory) {

  var root =
    (typeof self === 'object' && self.self === self && self) ||
    (typeof global === 'object' && global.global === global && global);

  if (typeof define === 'function' && define.amd) {
    define(['exports'], function (exports) {
      root.ParticleNetwork = factory(root, exports);
    });
  } else if (typeof module === 'object' && module.exports) {
    module.exports = factory(root, {});
  } else {
    root.ParticleNetwork = factory(root, {});
  }

}(function (root, ParticleNetwork) {

  // =========================
  // Particle class
  // =========================
  var Particle = function (parent) {

    this.canvas = parent.canvas;
    this.ctx = parent.ctx;

    // 🔹 CHANGE: support array of colors
    if (Array.isArray(parent.options.particleColor)) {
      this.particleColor =
        parent.options.particleColor[
          Math.floor(Math.random() * parent.options.particleColor.length)
        ];
    } else {
      this.particleColor = parent.options.particleColor;
    }

    this.x = Math.random() * this.canvas.width;
    this.y = Math.random() * this.canvas.height;
    this.velocity = {
      x: (Math.random() - 0.5) * parent.options.velocity,
      y: (Math.random() - 0.5) * parent.options.velocity
    };
  };

  Particle.prototype.update = function () {

    if (this.x > this.canvas.width + 20 || this.x < -20) {
      this.velocity.x = -this.velocity.x;
    }
    if (this.y > this.canvas.height + 20 || this.y < -20) {
      this.velocity.y = -this.velocity.y;
    }

    this.x += this.velocity.x;
    this.y += this.velocity.y;
  };

  Particle.prototype.draw = function () {

    this.ctx.beginPath();
    this.ctx.fillStyle = this.particleColor;
    this.ctx.globalAlpha = 0.7;
    this.ctx.arc(this.x, this.y, 1.5, 0, 2 * Math.PI);
    this.ctx.fill();
  };

  // =========================
  // ParticleNetwork class
  // =========================
  ParticleNetwork = function (canvas, options) {

    this.canvasDiv = canvas;
    this.canvasDiv.size = {
      width: this.canvasDiv.offsetWidth,
      height: this.canvasDiv.offsetHeight
    };

    options = options || {};

    this.options = {
      // 🔹 CHANGE: default is 4 colors
      particleColor: options.particleColor || [
        '#ff4d4d',
        '#4d79ff',
        '#4dff88',
        '#ffd24d'
      ],
      background: options.background || '#1a252f',
      interactive: options.interactive !== undefined ? options.interactive : true,
      velocity: this.setVelocity(options.speed),
      density: this.setDensity(options.density)
    };

    this.init();
  };

  ParticleNetwork.prototype.init = function () {

    this.bgDiv = document.createElement('div');
    this.canvasDiv.appendChild(this.bgDiv);
    this.setStyles(this.bgDiv, {
      position: 'absolute',
      top: 0,
      left: 0,
      bottom: 0,
      right: 0,
      'z-index': 1
    });

    if (
      (/(^#[0-9A-F]{6}$)|(^#[0-9A-F]{3}$)/i).test(this.options.background) ||
      this.options.background === 'transparent'
    ) {
      this.setStyles(this.bgDiv, {
        background: this.options.background
      });
    } else if ((/\.(gif|jpg|jpeg|tiff|png)$/i).test(this.options.background)) {
      this.setStyles(this.bgDiv, {
        background: 'url("' + this.options.background + '") no-repeat center',
        'background-size': 'cover'
      });
    } else {
      console.error('Please specify a valid background image or hex color');
      return false;
    }

    // 🔹 CHANGE: validate array OR single hex
    if (
      !Array.isArray(this.options.particleColor) &&
      !(/(^#[0-9A-F]{6}$)|(^#[0-9A-F]{3}$)/i).test(this.options.particleColor)
    ) {
      console.error('Please specify a valid particleColor (hex or array)');
      return false;
    }

    this.canvas = document.createElement('canvas');
    this.canvasDiv.appendChild(this.canvas);
    this.ctx = this.canvas.getContext('2d');

    this.canvas.width = this.canvasDiv.size.width;
    this.canvas.height = this.canvasDiv.size.height;

    this.setStyles(this.canvasDiv, { position: 'relative' });
    this.setStyles(this.canvas, {
      position: 'relative',
      'z-index': 20
    });

    window.addEventListener('resize', function () {

      if (
        this.canvasDiv.offsetWidth === this.canvasDiv.size.width &&
        this.canvasDiv.offsetHeight === this.canvasDiv.size.height
      ) return;

      this.canvas.width = this.canvasDiv.size.width = this.canvasDiv.offsetWidth;
      this.canvas.height = this.canvasDiv.size.height = this.canvasDiv.offsetHeight;

      clearTimeout(this.resetTimer);
      this.resetTimer = setTimeout(function () {

        this.particles = [];
        for (var i = 0; i < this.canvas.width * this.canvas.height / this.options.density; i++) {
          this.particles.push(new Particle(this));
        }

        if (this.options.interactive) {
          this.particles.push(this.mouseParticle);
        }

        requestAnimationFrame(this.update.bind(this));
      }.bind(this), 500);

    }.bind(this));

    this.particles = [];
    for (var i = 0; i < this.canvas.width * this.canvas.height / this.options.density; i++) {
      this.particles.push(new Particle(this));
    }

    if (this.options.interactive) {

      this.mouseParticle = new Particle(this);
      this.mouseParticle.velocity = { x: 0, y: 0 };
      this.particles.push(this.mouseParticle);

      this.canvas.addEventListener('mousemove', function (e) {
        this.mouseParticle.x = e.clientX - this.canvas.offsetLeft;
        this.mouseParticle.y = e.clientY - this.canvas.offsetTop;
      }.bind(this));

      this.canvas.addEventListener('mouseup', function () {
        this.mouseParticle.velocity = {
          x: (Math.random() - 0.5) * this.options.velocity,
          y: (Math.random() - 0.5) * this.options.velocity
        };
        this.mouseParticle = new Particle(this);
        this.mouseParticle.velocity = { x: 0, y: 0 };
        this.particles.push(this.mouseParticle);
      }.bind(this));
    }

    requestAnimationFrame(this.update.bind(this));
  };

  ParticleNetwork.prototype.update = function () {

    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.ctx.globalAlpha = 1;

    for (var i = 0; i < this.particles.length; i++) {

      this.particles[i].update();
      this.particles[i].draw();

      for (var j = this.particles.length - 1; j > i; j--) {

        var distance = Math.sqrt(
          Math.pow(this.particles[i].x - this.particles[j].x, 2) +
          Math.pow(this.particles[i].y - this.particles[j].y, 2)
        );

        if (distance > 120) continue;

        this.ctx.beginPath();
        // 🔹 CHANGE: connection follows particle color
        this.ctx.strokeStyle = this.particles[i].particleColor;
        this.ctx.globalAlpha = (120 - distance) / 120;
        this.ctx.lineWidth = 0.7;
        this.ctx.moveTo(this.particles[i].x, this.particles[i].y);
        this.ctx.lineTo(this.particles[j].x, this.particles[j].y);
        this.ctx.stroke();
      }
    }

    if (this.options.velocity !== 0) {
      requestAnimationFrame(this.update.bind(this));
    }
  };

  ParticleNetwork.prototype.setVelocity = function (speed) {
    if (speed === 'fast') return 1;
    if (speed === 'slow') return 0.33;
    if (speed === 'none') return 0;
    return 0.66;
  };

  ParticleNetwork.prototype.setDensity = function (density) {
    if (density === 'high') return 5000;
    if (density === 'low') return 20000;
    return !isNaN(parseInt(density, 10)) ? density : 10000;
  };

  ParticleNetwork.prototype.setStyles = function (div, styles) {
    for (var property in styles) {
      div.style[property] = styles[property];
    }
  };

  return ParticleNetwork;

}));
