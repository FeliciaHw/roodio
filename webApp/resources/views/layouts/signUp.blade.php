@extends('layouts.master')


@push('script')
    <script type="text/javascript" src="{{ asset('js/design/register-bg.js') }}" defer></script>
    <script type="text/javascript" src="{{ asset('js/design/particle-network.js') }}" defer></script>
@endpush


@section('bodyContent')
    <div id="particle-canvas" class='min-h-lvh h-fit relative z-1'></div>
    <div class='absolute z-10 w-full h-fit flex justify-center items-center top-6 pb-6'>
        <div class='border-primary-30 border-4 rounded-4xl bg-secondary-happy-10/85 w-sm h-max p-5 pt-5 font-secondaryAndButton shadow-xl shadow-primary-20/40 md:w-md lg:w-[30rem]'>
            <div class='flex flex-col items-center gap-1'>
                 @yield('mainImage')
                <p class='font-primary text-subtitle font-bold text-primary-85'>
                    @yield('mainTitle')
                </p>
                <p class='text-center text-primary-50 text-small mb-3'>
                    @yield('description')
                </p>
            </div>
            <div>
                @yield('content')
            </div>
        </div>
    </div>
@endsection
