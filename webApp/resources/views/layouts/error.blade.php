@extends('layouts.master')


@php
    if (!isset($pageTitle)) {
        $pageTitle = $errorCode . ' ' . $errorIdentity;
    };
@endphp


@section('title', $pageTitle)


@section('bodyClass', 'flex items-center')


@section('bodyContent')
    <div class='flex flex-col w-full h-full items-center justify-center mx-5 lg:flex-row'>
        <div class='w-2xs mb-1 mx-auto lg:w-xs lg:mx-0 lg:mb-0'>
            <img src="{{ asset("assets/errors/". $errorCode .".png") }}" alt="{{ $errorCode }}">
        </div>
        <div class='font-secondaryAndButton text-small text-white text-center lg:text-justify lg:ml-8 lg:text-body-size'>
            <p class='font-primary font-bold text-warning-lighten text-body-size mb-1 lg:text-paragraph'>{{ $errorCode }} | {{ $errorIdentity }}</p>
            <p>{{ $errorDescription }}</p>
        </div>
    </div>
@endsection