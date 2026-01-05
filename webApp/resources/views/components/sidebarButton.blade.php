@props([
    'route' => 'welcome',
    'icon' => 'home',
    'isToggle' => true,
    'mood' => 'happy',
    'isActive' => false,
    'label' => 'label',
    'content' => 'this is content'
])


@php
    $isActive = request()->routeIs($route);

    $iconContainerStyle = [
        'happy' => 'group-hover:bg-secondary-happy-10 group-hover:border-y-secondary-happy-100',
        'sad' => 'group-hover:bg-secondary-sad-10 group-hover:border-y-secondary-sad-100',
        'relaxed' => 'group-hover:bg-secondary-relaxed-10 group-hover:border-y-secondary-relaxed-100',
        'angry' => 'group-hover:bg-secondary-angry-10 group-hover:border-y-secondary-angry-100'
    ];

    $iconContainerActiveStyle = [
        'happy' => 'bg-secondary-happy-10 border-r-secondary-happy-85',
        'sad' => 'bg-secondary-sad-10 border-r-secondary-sad-85',
        'relaxed' => 'bg-secondary-relaxed-10 border-r-secondary-relaxed-85',
        'angry' => 'bg-secondary-angry-10 border-r-secondary-angry-85'
    ];

    $iconStyle = [
        'happy' => 'bg-secondary-happy-10 group-hover:bg-secondary-happy-60',
        'sad' => 'bg-secondary-sad-10 group-hover:bg-secondary-sad-60',
        'relaxed' => 'bg-secondary-relaxed-10 group-hover:bg-secondary-relaxed-60',
        'angry' => 'bg-secondary-angry-10 group-hover:bg-secondary-angry-60'
    ];

    $labelActiveStyle = [
        'happy' => 'text-secondary-happy-100',
        'sad' => 'text-secondary-sad-100',
        'relaxed' => 'text-secondary-relaxed-100',
        'angry' => 'text-secondary-angry-100'
    ];

    $musicDiscStyle = [
        'happy' => 'from-secondary-happy-50 via-white via-30% to-secondary-happy-50 to-60%',
        'sad' => 'from-secondary-sad-50 via-white via-30% to-secondary-sad-50 to-60%',
        'relaxed' => 'from-secondary-relaxed-50 via-white via-30% to-secondary-relaxed-50 to-60%',
        'angry' => 'from-secondary-angry-50 via-white via-30% to-secondary-angry-50 to-60%'
    ];

    $contentStyle = [
        'happy' => 'from-secondary-happy-100 to-secondary-happy-20',
        'sad' => 'from-secondary-sad-100 to-secondary-sad-20',
        'relaxed' => 'from-secondary-relaxed-100 to-secondary-relaxed-20',
        'angry' => 'from-secondary-angry-100 to-secondary-angry-20'
    ];
    
    $backgroundToggleStyle = [
        'happy' => 'bg-secondary-happy-20 group-hover:bg-secondary-happy-30',
        'sad' => 'bg-secondary-sad-20 group-hover:bg-secondary-sad-30',
        'relaxed' => 'bg-secondary-relaxed-20 group-hover:bg-secondary-relaxed-30',
        'angry' => 'bg-secondary-angry-20 group-hover:bg-secondary-angry-30'
    ];

    $backgroundToggleActiveStyle = [
        'happy' => 'bg-secondary-happy-50',
        'sad' => 'bg-secondary-sad-50',
        'relaxed' => 'bg-secondary-relaxed-50',
        'angry' => 'bg-secondary-angry-50'
    ];

    $labelToggleStyle = [
        'happy' => 'bg-secondary-happy-10/90 group-hover:bg-secondary-happy-20',
        'sad' => 'bg-secondary-sad-10/90 group-hover:bg-secondary-sad-20',
        'relaxed' => 'bg-secondary-relaxed-10/90 group-hover:bg-secondary-relaxed-20',
        'angry' => 'bg-secondary-angry-10/90 group-hover:bg-secondary-angry-20'
    ];

    $labelToggleActiveStyle = [
        'happy' => 'to-secondary-happy-60 from-secondary-happy-20',
        'sad' => 'to-secondary-sad-60 from-secondary-sad-20',
        'relaxed' => 'to-secondary-relaxed-60 from-secondary-relaxed-20',
        'angry' => 'to-secondary-angry-60 from-secondary-angry-20'
    ];
@endphp


<a href="{{ route($route) }}" class="{{ ($isActive) ? '' : 'group' }} w-fit inline-block " style="{{ ($isActive) ? 'pointer-events: none;' : '' }}">
    @if (!$isToggle)
        <div class="relative font-secondaryAndButton">
            <div
                {{ $attributes->merge([
                    'class' => 'w-18 h-18 p-3 relative z-10 flex flex-col items-center justify-center group-hover:border-y-2 duration-100 lg:w-20 lg:h-20 ' . $iconContainerStyle[$mood] . ' ' . (($isActive) ? $iconContainerActiveStyle[$mood] . ' border-r-4' : ' bg-primary-85')
                ]) }}
            >
                <div 
                {{ $attributes->merge([
                    'class' => 'w-10 p-2 rounded-full ' . $iconStyle[$mood] . ' '
                ])
                }}
                >
                    <img src="{{ asset('assets/icons/'. $icon .'.svg') }}" alt="{{ $icon }}">
                </div>
                <p
                {{ 
                    $attributes->merge([
                        "class" => 'text-micro group-hover:text-primary-70 group-hover:font-bold lg:text-small ' . (($isActive) ? $labelActiveStyle[$mood] . ' font-bold' : 'text-white') . ' '
                    ])
                }}
                >{{ $label }}</p>
            </div>
            <div
            {{ 
                $attributes->merge([
                    'class' => 'flex items-center justify-center w-16 h-16 rounded-full absolute z-5 top-1/2 left-0 -translate-y-1/2 group-hover:translate-x-3/5 group-hover:animate-spin duration-350 transition-transform bg-conic border border-primary-50 lg:w-18 lg:h-18 '. $musicDiscStyle[$mood] . ' '
                ])
            }}
            >
                <img src="{{ asset('assets/logo-no-text.png') }}" alt="logo" class='w-9 p-1 bg-primary-70 rounded-full lg:w-10'>
            </div>
            <div 
            {{ 
                $attributes->merge([
                    'class' => 'bg-linear-to-r w-max h-max rounded-md pl-29 px-3 py-1 absolute z-3 top-1/2 left-0 -translate-x-full -translate-y-1/2 transition-transform duration-350 group-hover:translate-x-0 lg:pl-33 ' . $contentStyle[$mood] . ' '
                ])
            }}
            >
                <p>{{ $content }}</p>
            </div>
        </div>
    @else
        <div class='relative w-max font-secondaryAndButton'>
            <div class='h-max flex flex-row relative'>
                <div
                {{ 
                    $attributes->merge([
                        "class" => 'p-2 w-fit relative z-5 rounded-l-xl border-r-4 border-r-primary-50 ' . ' group-hover:' . $backgroundToggleActiveStyle[$mood] . ' ' . (($isActive) ? $backgroundToggleActiveStyle[$mood] : $backgroundToggleStyle[$mood]) . ' '
                    ])
                }}
                >
                    <img src="{{ asset('assets/icons/'. $icon .'.svg') }}" alt="{{ $icon }}" class='w-8 h-8'>
                </div>
                <div
                {{ 
                    $attributes->merge([
                        "class" => 'w-36 relative z-3 px-8 flex flex-col justify-center rounded-r-xl ' . $labelToggleStyle[$mood] . ' ' . (($isActive) ? 'bg-linear-to-r ' . $labelToggleActiveStyle[$mood] : '') . ' '
                    ])
                }}
                >
                    <p
                    {{ 
                        $attributes->merge([
                            "class" => 'text-small text-primary-60 ' . (($isActive) ? 'font-bold' : '') . ' '
                        ])
                    }}
                    >{{ $label }}</p>
                </div>
            </div>
            <div
            {{ 
                $attributes->merge([
                    'class' => 'absolute z-4 w-12 h-12 rounded-full left-1/8 top-1/2 -translate-y-1/2 flex items-center justify-center rounded-full group-hover:animate-spin duration-350 transition-transform bg-conic border border-primary-50 '. $musicDiscStyle[$mood] . ' ' . (($isActive) ? 'animate-spin' : '') . ' '
                ])
            }}
            >
                <img src="{{ asset('assets/logo-no-text.png') }}" alt="logo" class='w-6 p-1 bg-primary-70 rounded-full'>
            </div>
        </div>
    @endif
</a>