<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Create Song</title>
    @vite(['resources/css/app.css', 'resources/js/app.js'])
</head>
<body>
    <form action="{{ route('admin.songs.store') }}" method="POST" enctype="multipart/form-data">
        @csrf
        <label for="title">Title</label>
        <input type="text" name="title">
        @error('title')
            <div class="invalid-feedback">
                {{ $message }}
            </div>
        @enderror
        <label for="lyrics">Lyrics</label>
        <textarea name="lyrics"></textarea>
        @error('lyrics')
            <div class="invalid-feedback">
                {{ $message }}
            </div>
        @enderror
        <label for="artist">Artist</label>
        <input type="text" name="artist">
        @error('artist')
            <div class="invalid-feedback">
                {{ $message }}
            </div>
        @enderror
        <label for="genre">genre</label>
        <input type="text" name="genre">
        @error('genre')
            <div class="invalid-feedback">
                {{ $message }}
            </div>
        @enderror
        <label for="publisher">publisher</label>
        <input type="text" name="publisher">
        @error('publisher')
            <div class="invalid-feedback">
                {{ $message }}
            </div>
        @enderror
        <label for="datePublished"></label>
        <input type="date" name="datePublished">
        @error('datePublished')
            <div class="invalid-feedback">
                {{ $message }}
            </div>
        @enderror
        <label name="song">
        <input type="file" name="song">
        @error('song')
            <div class="invalid-feedback">
                {{ $message }}
            </div>
        @enderror
        <button type="submit">Submit</button>
    </form>
</body>
</html>
