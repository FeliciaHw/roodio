@props([
    'type' => 'text',
    'id',
    'name' => null,
    'icon' => null,
    'label' => null,
    'additionalInfo' => null,
    'isRequired' => true,
    'placeholder' => null
])

@php
    $name = $name ?? $id;

    $baseStyle = 'w-full px-1.5 py-0.5 text-small border-b-2 rounded-md border-shadedOfGray-50 not-placeholder-shown:bg-accent-20/60 not-placeholder-shown:text-shadedOfGray-100 placeholder:text-micro placeholder:italic focus:border-secondary-happy-100 focus:border-b-2 focus:bg-secondary-happy-20/65 hover:bg-shadedOfGray-30/45 ease-in-out duration-125 md:text-body-size md:placeholder:text-small md:h-9';
    $errorStyle = ($errors->has($name)) ? 'bg-error-lighten/25 border-error-dark' : 'bg-shadedOfGray-20/50';

    $typeStyle = '';
    switch ($type) {
        case 'password':
            $typeStyle = 'pr-8';
            break;

        default:
            $typeStyle = '';
            break;
    }
@endphp

<div class='flex flex-col p-1 h-max mb-5'>
    <label for="{{ $name }}" class='text-small flex flex-row mb-1 md:text-body-size'>
        @isset($icon)
            <img src="{{ asset('assets/icons/'. $icon .'.svg') }}" alt='{{ $icon }}' class='w-5 mr-1.5 md:w-6 lg:w-7'>
        @endisset

        @isset($label)
            <p class='text-primary-85'>
                {{ $label }}@if ($isRequired)<span class='text-error-dark'>*</span>@endif
            </p>
        @endisset
    </label>

    @isset($additionalInfo)
        <p class='text-micro text-shadedOfGray-70 font-bold md:text-small'>{{ $additionalInfo }}</p>
    @endisset

    <div class='relative'>
        <input
            type="{{ $type }}"
            id="{{ $id }}"
            name="{{ $name }}"
            autocomplete="{{ ($attributes->has('autocomplete')) ? 'on' : 'off' }}"
            @isset($placeholder)
                placeholder="{{ $placeholder }}"
            @endisset
            {{ $attributes->merge([
                'class' => $baseStyle . ' ' . $typeStyle . ' ' . $errorStyle
            ]) }}
        />
        @isset($additionalContent)
            {{ $additionalContent }}
        @endisset
    </div>

    <div class="error-message">
        @error($name)
            {{ $message }}
        @enderror
    </div>
</div>
