@extends('layouts.signUp')


@section('title', 'ROODIO - Account')


@push('script')
    <script src="{{ asset('js/auth/password.js') }}" defer></script>
@endpush


@section('mainImage')
    <img src="{{ asset('assets/logo-with-text.png') }}" alt="logo" class='w-24 drop-shadow-sm drop-shadow-primary-50'>
@endsection


@section('mainTitle', 'USER ACCOUNT')


@section('description', "Awesome! One step left with ROODIO!")


@section('content')
    <form action="{{ route('auth.account') }}" method="POST" id='account'>
        @csrf
        <div class='flex flex-col mb-6'>
            <label for="username" class='text-body-size flex flex-row mb-1'>
                <img src="{{ asset('assets/icons/user.svg') }}" alt="username" class='w-6 mr-1'>
                <p class='text-primary-85 text-body-size'>Your Username<span class='text-danger'>*</span></p>
            </label>
            <p class='text-small text-shadedOfGray-70 font-bold'>You can customize your username.</p>
            <input type="text" name="username" id="username" autocomplete="off" placeholder="Create your username..." value="user321_" class='not-placeholder-shown:bg-accent-20/60 not-placeholder-shown:text-shadedOfGray-100 text-small outline-none border-b rounded-md px-1.5 py-0.5 border-shadedOfGray-50 placeholder:text-micro placeholder:italic focus:border-secondary-happy-100 focus:border-b-2 focus:bg-secondary-happy-20/50 ease-in-out duration-150 hover:bg-shadedOfGray-20/90 md:text-body-size md:h-8 md:placeholder:text-small {{ $errors->has('username') ? 'border-error-dark border-b-2 bg-error-lighten/30' : 'border-shadedOfGray-50' }}'>
            <div class="error-message">
                @error('username')
                    {{ $message }}
                @enderror
            </div>
        </div>
        <div class='flex flex-col mb-6'>
            <label for="password" class='text-body-size flex flex-row mb-2'>
                <img src="{{ asset('assets/icons/password.svg') }}" alt="password" class='w-6 mr-1'>
                <p class='text-primary-85 text-body-size'>Your Password<span class='text-danger'>*</span></p>
            </label>
            <div class='relative'>
                <input type="password" name="password" id="password" autocomplete="off" placeholder="Create your password..." class='not-placeholder-shown:bg-accent-20/60 not-placeholder-shown:text-shadedOfGray-100 text-small outline-none border-b rounded-md px-1.5 py-0.5 border-shadedOfGray-50 placeholder:text-micro focus:border-secondary-happy-100 focus:border-b-2 focus:bg-secondary-happy-20/50 ease-in-out duration-150 hover:bg-shadedOfGray-20/90 md:text-body-size md:h-8 md:placeholder:text-small pr-8 w-full {{ $errors->has('password') ? 'border-error-dark border-b-2 bg-error-lighten/30' : 'border-shadedOfGray-50' }}'>
                <button type='button' id='showPass' class='w-4 h-4 absolute z-4 right-1.5 bottom-1.5 flex items-center justify-center cursor-pointer md:bottom-2 md:right-2'>
                    <img src="{{ asset('assets/icons/eye-closed.svg') }}" alt="eye-closed" id='eye-closed'>
                    <span class='absolute invisible' id='eye-open'>&#128065;</span>
                </button>
            </div>
<<<<<<< HEAD
            <div class="error-message">
=======
            <div class="text-error-moderate h-3 pt-0.5 text-micro md:text-small">
>>>>>>> 51dadadbc50128a6ab81f4d7091672c9353fd4e3
                @error('password')
                    {{ $message }}
                @enderror
            </div>
        </div>
        <div class='flex flex-col mb-8'>
            <label for="passwordConfirm" class='text-body-size flex flex-row mb-2'>
                <img src="{{ asset('assets/icons/password.svg') }}" alt="password" class='w-6 mr-1'>
                <p class='text-primary-85 text-body-size'>Password Confirmation<span class='text-danger'>*</span></p>
            </label>
            <div class='relative'>
                <input type="password" name="passwordConfirm" id="passwordConfirm" autocomplete="off" placeholder="Input again your password..." class='not-placeholder-shown:bg-accent-20/60 not-placeholder-shown:text-shadedOfGray-100 text-small outline-none border-b rounded-md px-1.5 py-0.5 border-shadedOfGray-50 placeholder:text-micro focus:border-secondary-happy-100 focus:border-b-2 focus:bg-secondary-happy-20/50 ease-in-out duration-150 hover:bg-shadedOfGray-20/90 md:text-body-size md:h-8 md:placeholder:text-small pr-8 w-full {{ $errors->has('passwordConfirm') ? 'border-error-dark border-b-2 bg-error-lighten/30' : 'border-shadedOfGray-50' }}'>
                <button type='button' id='showPassConfirm' class='w-4 h-4 absolute z-4 right-1.5 bottom-1.5 flex items-center justify-center cursor-pointer md:bottom-2 md:right-2'>
                    <img src="{{ asset('assets/icons/eye-closed.svg') }}" alt="eye-closed" id='eye-closed-pass-confirm'>
                    <span class='absolute invisible' id='eye-open-pass-confirm'>&#128065;</span>
                </button>
            </div>
<<<<<<< HEAD
            <div class="error-message">
=======
            <div class="text-error-moderate h-3 pt-0.5 text-micro md:text-small">
>>>>>>> 51dadadbc50128a6ab81f4d7091672c9353fd4e3
                @error('passwordConfirm')
                    {{ $message }}
                @enderror
            </div>
        </div>
        <div>
            <button type='submit' form='account' class='text-smallBtn font-bold w-full font-secondaryAndButton bg-primary-10 text-primary-100 rounded-2xl py-1 mb-2 cursor-pointer hover:bg-primary-50 hover:text-white ease-in-out duration-150'>Create Account</button>
        </div>
    </form>
@endsection