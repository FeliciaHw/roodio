<?php
namespace App\Providers;

use Illuminate\Filesystem\FilesystemAdapter;
use Illuminate\Support\Facades\Storage;
use Illuminate\Support\ServiceProvider;
use League\Flysystem\AzureBlobStorage\AzureBlobStorageAdapter;
use League\Flysystem\Filesystem;
use MicrosoftAzure\Storage\Blob\BlobRestProxy;

class AzureFilesystemServiceProvider extends ServiceProvider
{
    public function boot(): void
    {
        Storage::extend('azure', function ($app, $config) {

            $connectionString = sprintf(
                'DefaultEndpointsProtocol=https;AccountName=%s;AccountKey=%s;EndpointSuffix=core.windows.net',
                $config['name'],
                $config['key']
            );

            $client = BlobRestProxy::createBlobService($connectionString);

            $adapter = new AzureBlobStorageAdapter(
                $client,
                $config['container']
            );

            $filesystem = new Filesystem($adapter);

            // 🔑 INI YANG PENTING
            return new FilesystemAdapter(
                $filesystem,
                $adapter,
                $config
            );
        });
    }
}
