@extends('layouts/master')


@section('title', 'ROODIO - Login')


@push('script')
    <script type="text/javascript" src="{{ asset('js/design/login-bg.js') }}" defer></script>
    <script type="text/javascript" src="{{ asset('js/design/particle-network.js') }}" defer></script>
    <script type="text/javascript" src="{{ asset('js/auth/password.js') }}" defer></script>
@endpush


@section('bodyContent')
    <div class='grid w-screen h-screen bg-primary-100 overflow-hidden justify-items-center items-center mx-auto'>
        <div id="particle-canvas"></div>
        <div class='w-xs h-[26rem] border-secondary-happy-50 border-3 bg-white/75 hover:bg-white/85 transition-colors active:bg-white/85 ease-linear duration-150 rounded-4xl col-start-1 row-start-1 z-10 flex flex-col items-center justify-around shadow-xl shadow-secondary-happy-50/40 md:w-md md:h-[32rem]'>
            <div class='w-[inherit] animate-pulse mt-4 flex flex-row-max h-max items-center justify-center font-primary text-title font-bold text-primary-85 md:text-hero'>
                <span>L</span>
                <div class='rounded-full w-12 h-12 bg-primary-85 mx-2 flex justify-center items-center md:w-16 md:h-16'>
                    <img src="{{ asset('assets/logo-no-text.png') }}" alt="logo" class='w-5 animate-spin bg-amber-200 rounded-full drop-shadow-md/70 drop-shadow-white/70 md:w-7'>
                </div>
                <span class='tracking-[.5rem]'>GI</span>
                <span>N</span>
            </div>
            <p class='font-secondaryAndButton -mt-3 text-small text-primary-50 md:text-body-size md:-mt-9 md:tracking-[.05rem]'>Welcome back to ROODIO!</p>
            <div class='font-secondaryAndButton text-body-size flex justify-center w-75 h-max'>
                <form action="{{ route('auth.login') }}" method='POST' id='login'>
                    @csrf {{-- cross site request forgery --}}
                    <div class='flex flex-col w-3xs mb-5 md:w-sm md:mb-8'>
                        <label for="username" class='mb-1 text-primary-70 md:mb-1.5'>
                            <img src="{{ asset('assets/icons/user.svg') }}" alt="user" class='w-5 inline'>
                            <p class='align-middle inline'>Username</p>
                        </label>
                        <input type="text" name="username" id="username" autocomplete="off" value="{{ old('username') }}" placeholder="Input your username..." class='not-placeholder-shown:bg-accent-20/60 not-placeholder-shown:text-shadedOfGray-100 text-small outline-none border-b rounded-md px-1.5 py-0.5 placeholder:text-micro focus:border-secondary-happy-100 focus:border-b-2 focus:bg-secondary-happy-20/50 ease-in-out duration-150 hover:bg-shadedOfGray-20/90 md:text-body-size md:h-8 md:placeholder:text-small {{ $errors->has('username') ? 'border-error-dark border-b-2 bg-error-lighten/30' : 'border-shadedOfGray-50' }}'>
                        <div class="text-error-moderate text-xs h-3 pt-0.5 md:text-small">
                            @error('username')
                                {{ $message }}
                            @enderror
                        </div>
                    </div>
                    <div class='flex flex-col w-3xs relative mb-7 md:w-sm md:mb-9'>
                        <label for="password" class='mb-1 text-primary-70 relative'>
                            <img src="{{ asset('assets/icons/password.svg') }}" alt="password" class='w-5 inline'>
                            <p class='align-middle inline'>
                                Password
                                <a href='/forget-password' class='right-0 top-1/2 -translate-y-1/2 absolute text-xs font-bold text-secondary-sad-100 md:text-micro hover:text-primary-50'>Forget Password?</a>
                            </p>

                        </label>
                        <div class='relative'>
                            <input type="password" name="password" id="password" autocomplete="off" placeholder="Input your password..." class='not-placeholder-shown:bg-accent-20/60 not-placeholder-shown:text-shadedOfGray-100 text-small outline-none border-b rounded-md px-1.5 py-0.5 w-full placeholder:text-micro focus:border-secondary-happy-100 focus:border-b-2 focus:bg-secondary-happy-20/50 ease-in-out duration-150 hover:bg-shadedOfGray-20/90 md:text-body-size md:h-8 md:placeholder:text-small pr-8 {{ $errors->has('password') ? 'border-error-dark border-b-2 bg-error-lighten/30' : 'border-shadedOfGray-50' }}'>
                            <button type='button' id='showPass' class='w-4 h-4 absolute right-1.5 bottom-1.5 flex items-center justify-center cursor-pointer md:bottom-2 md:right-2'>
                                <img src="{{ asset('assets/icons/eye-closed.svg') }}" alt="eye-closed" id='eye-closed'>
                                <span class='absolute invisible' id='eye-open'>&#128065;</span>
                            </button>
                        </div>
                        <div class="text-error-moderate text-xs h-3 pt-0.5 md:text-small">
                            @error('password')
                                {{ $message }}
                            @enderror
                        </div>
                    </div>
                    <div class='mb-4'>
                        <button type="submit" name='login' form='login' class='text-smallBtn font-bold w-3xs font-secondaryAndButton bg-primary-10 text-primary-100 rounded-2xl py-1 mb-2 cursor-pointer hover:bg-primary-50 hover:text-white ease-in-out duration-150 md:w-sm md:h-11'>Login</button>
                        <p class='text-micro text-center md:text-small'>Don't have account? <a href="/sign-up" class='font-bold text-secondary-sad-100 hover:text-primary-50'>Sign Up Here!</a></p>
                    </div>
                </form>
            </div>
        </div>
    </div>
@endsection