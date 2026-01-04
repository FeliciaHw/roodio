@props([
    'route' => 'welcome',
    'icon' => 'home',
    'isToggle' => false,
    'mood' => "happy",
    'isActive' => false,
    'label' => 'label',
    'content' => 'this is content'
])


@php
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
        'happy' => 'bg-secondary-happy-10 hover:bg-secondary-happy-70',
        'sad' => 'bg-secondary-sad-10 hover:bg-secondary-sad-70',
        'relaxed' => 'bg-secondary-relaxed-10 hover:bg-secondary-relaxed-70',
        'angry' => 'bg-secondary-angry-10 hover:bg-secondary-angry-70'
    ];

    $iconToggleStyle = [
        'happy' => 'group-hover:bg-secondary-happy-10',
        'sad' => 'group-hover:bg-secondary-sad-10',
        'relaxed' => 'group-hover:bg-secondary-relaxed-10',
        'angry' => 'group-hover:bg-secondary-angry-10'
    ];


    $backgroundToggleActiveStyle = [
        'happy' => 'border-2 border-secondary-happy-85 bg-secondary-happy-20',
        'sad' => 'border-2 border-secondary-sad-85 bg-secondary-sad-20',
        'relaxed' => 'border-2 border-secondary-relaxed-85 bg-secondary-relaxed-20',
        'angry' => 'border-2 border-secondary-angry-85 bg-secondary-angry-20'
    ];
@endphp


<a href="{{ route($route) }}" class="{{ ($isActive) ? '' : 'group' }} inline-block" style="{{ ($isActive) ? 'pointer-events: none;' : '' }}">
    @if (!$isToggle)
        <div class="relative font-secondaryAndButton w-fit">
            <div
                {{ $attributes->merge([
                    'class' => 'w-18 h-18 p-3 relative z-10 flex flex-col items-center justify-center group-hover:border-y duration-100 lg:w-20 lg:h-20 ' . $iconContainerStyle[$mood] . ' ' . (($isActive) ? $iconContainerActiveStyle[$mood] . ' border-r-4' : ' bg-primary-85')
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
        <div
        {{ 
            $attributes->merge([
                "class" => 'rounded-md w-40 py-2 px-4 overflow-hidden flex flex-row items-center justify-start gap-3 font-secondaryAndButton duration-125 lg:w-44 ' . (($isActive) ? $backgroundToggleActiveStyle[$mood] : $backgroundToggleStyle[$mood]) . ' '
            ])
        }}
        >
            <div
            {{ 
                $attributes->merge([
                    "class" => 'w-10 h-10 p-1.5 group-hover:rounded-full ' . $iconToggleStyle[$mood] . ' '
                ])
            }}
            >
                <img src="{{ asset('assets/icons/'. $icon .'.svg') }}" alt="{{ $icon }}">
            </div>
            <div
            {{ 
                $attributes->merge([
                    "class" => 'w-max font-secondaryAndButton group-hover:text-white group-hover:font-bold text-smallBtn lg:text-mediumBtn ' . (($isActive) ? 'font-bold text-primary-50' : $labelActiveStyle[$mood]) . ' '
                ])
            }}
            >
                <p>{{ $label }}</p>
            </div>
        </div>
    @endif
</a>