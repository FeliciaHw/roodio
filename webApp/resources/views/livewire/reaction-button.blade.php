<div>
    <button
        wire:click="toggle"
        class="flex items-center gap-2">

        <span class="{{ $reacted ? 'text-red-500' : 'text-gray-400' }}">
            ❤️
        </span>

        <span>{{ $count }}</span>
    </button>
</div>
