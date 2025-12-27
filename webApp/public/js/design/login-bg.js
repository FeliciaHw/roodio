document.addEventListener('DOMContentLoaded', function () {
    const canvasDiv = document.getElementById('particle-canvas');

    if (!canvasDiv) {
        console.error('particle-canvas not found');
        return;
    }

    canvasDiv.style.width = 'inherit';
    canvasDiv.style.height = 'inherit';
    canvasDiv.classList.add('col-start-1', 'row-start-1', 'z-1');

    const options = {
        particleColor: ['#FFC48D', '#B6A5E7', '#8EE0B1', '#F49DA0'],
        background: 'transparent',
        interactive: true,
        speed: 'fast',
        density: 'medium',
    };

    new ParticleNetwork(canvasDiv, options);
});