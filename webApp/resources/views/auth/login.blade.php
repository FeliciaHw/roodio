@extends('layouts.signIn')


@section('title', 'ROODIO - Login')


@push('script')
    <script type="text/javascript" src="{{ asset('js/auth/password.js') }}" defer></script>
@endpush


@section('headContent')
    <div class='w-full flex flex-row h-max items-center justify-center font-primary text-title font-bold text-primary-70 md:text-hero animate-pulse'>
        <span>L</span>
        <div class='rounded-full w-12 h-12 bg-[conic-gradient(from_0deg,#7591DB_0%,#1F3A98_25%,#7591DB_50%,#1F3A98_75%,#7591DB_100%)] flex justify-center items-center mx-2 animate-spin md:w-16 md:h-16 border-2 border-primary-70'>
            <img src="{{ asset('assets/logo-no-text.png') }}" alt="logo" class='w-6 bg-secondary-happy-60 rounded-full md:w-8'>
        </div>
        <span class='tracking-[.5rem]'>GI</span>
        <span>N</span>
    </div>
    <p class='mb-3 font-secondaryAndButton text-small text-primary-50 md:text-body-size md:tracking-[.05rem]'>Welcome back to ROODIO!</p>
@endsection


@section('content')
    <form action="{{ route('auth.login') }}" method='POST' id='login'>
        @csrf {{-- cross site request forgery --}}
        <x-input id='username' icon='user' label='Username' placeholder='Input your username...' value="{{ old('username') }}"></x-input>
        <x-input type='password' id='password' icon='password' label='Password' placeholder='Input your password...'>
            <x-slot:inlineContent>
                <x-button behaviour='navigation' navType='text' :navLink="route('user.verification')" content="Forget Password?" class='w-fit inline'></x-button>
            </x-slot:inlineContent>
            <x-slot:additionalContent>
                <button type='button' id='showPass' class='w-4 h-4 absolute z-4 right-2.5 bottom-2 flex items-center justify-center cursor-pointer md:bottom-2.5'>
                    <img src="{{ asset('assets/icons/eye-closed.svg') }}" alt="eye-closed" id='eye-closed'>
                    <span class='absolute invisible' id='eye-open'>&#128065;</span>
                </button>
            </x-slot:additionalContent>
        </x-input>
        <div class='pt-2'>
            <x-button behaviour='action' actionType='submit' form='login' content='Login' class='min-w-full'></x-button>
            <div class='flex flex-row justify-center gap-1'>
                <p class='text-micro text-center md:text-small'>Don't have account?</p>
                <x-button behaviour='navigation' navType='text' :navLink="route('register')" content="Sign Up Here!"></x-button>
            </div>
        </div>
    </form>
@endsection