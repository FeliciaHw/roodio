@extends('layouts.master')


@push('script')
    <script type="text/javascript" src="{{ asset('js/design/register-bg.js') }}" defer></script>
    <script type="text/javascript" src="{{ asset('js/design/particle-network.js') }}" defer></script>
@endpush


@section('bodyContent')
    <div class='relative w-screen min-h-screen justify-items-center items-center'>
        <div id="particle-canvas" class='h-screen md:h-[130%] xl:h-full'></div>
        <div class='absolute z-10 border-primary-30 border-4 rounded-4xl bg-secondary-happy-10/85 w-sm h-max top-1/2 -translate-y-1/2 p-8 pt-5 font-secondaryAndButton shadow-xl shadow-primary-20/40 lg:w-md'>
            <div class='flex flex-col items-center gap-1'>
                @yield('mainImage')
                <p class='font-primary text-subtitle font-bold text-primary-85'>
                    @yield('mainTitle')
                </p>
                <p class='text-center text-primary-50 text-small mb-4'>
                    @yield('description')
                </p>
            </div>
            <div>
                @yield('content')
            </div>
        </div>
    </div>
@endsection